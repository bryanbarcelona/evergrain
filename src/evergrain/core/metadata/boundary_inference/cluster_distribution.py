from datetime import datetime, timedelta
from typing import cast

from evergrain.core.metadata.boundary_inference.event_scene_bounds import span_of
from evergrain.core.metadata.boundary_inference.row_classification import (
    classify_row,
    get_partial_date_bounds,
    get_row_time_bounds,
)
from evergrain.core.models.metadata import MetadataRow

_TIGHT_CLUSTER_SPREAD = timedelta(seconds=30)
_LOOSE_CLUSTER_SPREAD = timedelta(minutes=4)
_SINGLETON_SPREAD = timedelta(minutes=4)


def _get_cluster_spread_from_name(cluster_name: str | None, available: timedelta) -> timedelta:
    """Get cluster spread based on name."""
    if cluster_name and 'tight' in cluster_name.lower():
        return _TIGHT_CLUSTER_SPREAD
    if cluster_name and 'loose' in cluster_name.lower():
        return _LOOSE_CLUSTER_SPREAD
    return available


def _get_cluster_bounds_no_anchors(
    cluster_name: str | None,
    scene_start: datetime,
    scene_end: datetime,
) -> tuple[datetime, datetime]:
    """Fallback cluster bounds when no anchors are available — centers within the scene."""
    default_spread = _TIGHT_CLUSTER_SPREAD if cluster_name and 'tight' in cluster_name.lower() else _SINGLETON_SPREAD
    available_span = scene_end - scene_start
    actual_spread = min(default_spread, available_span)
    scene_center = scene_start + available_span / 2
    cluster_start = max(scene_start, scene_center - actual_spread / 2)
    cluster_end = min(scene_end, scene_center + actual_spread / 2)
    return cluster_start, cluster_end


def get_cluster_bounds(
    cluster_rows: list[MetadataRow], scene_start: datetime, scene_end: datetime
) -> tuple[datetime, datetime]:
    """Calculate cluster bounds with spread rules."""
    specific_anchors: list[tuple[datetime, datetime]] = []
    broad_anchors: list[tuple[datetime, datetime]] = []

    for r in cluster_rows:
        status = classify_row(r)
        if status in {'strong_anchor', 'partial_time'}:
            start, end = get_row_time_bounds(r)
            if start and end:
                specific_anchors.append((start, end))
        elif status == 'date_only':
            broad_anchors.append((
                datetime(r.year, r.month, r.day, 0, 0, 0),  # ty: ignore[invalid-argument-type]
                datetime(r.year, r.month, r.day, 23, 59, 59),  # ty: ignore[invalid-argument-type]
            ))
        elif status in {'year_month', 'year_only'}:
            start, end = get_partial_date_bounds(r)
            if start and end:
                broad_anchors.append((start, end))

    anchors = specific_anchors if specific_anchors else broad_anchors
    cluster_name = cluster_rows[0].Cluster if cluster_rows else None

    if not anchors:
        return _get_cluster_bounds_no_anchors(cluster_name, scene_start, scene_end)

    allowed_start, allowed_end = span_of(anchors)
    available = allowed_end - allowed_start
    spread = _get_cluster_spread_from_name(cluster_name, available)
    if spread > available:
        return allowed_start, allowed_end
    center = allowed_start + available / 2
    return center - spread / 2, center + spread / 2


def _build_anchor_positions(
    sorted_clusters: list[tuple[str, list[MetadataRow], bool, timedelta]],
) -> list[tuple[int, datetime, datetime]]:
    """Build sorted list of (min_row, anchor_start, anchor_end) for anchored clusters."""
    anchor_positions = []
    for name, rows, has_anchors, _ in sorted_clusters:
        if not has_anchors or name.startswith('__SINGLETON_'):
            continue
        min_row = min(r.row_num for r in rows)
        anchor_start = anchor_end = None
        for r in rows:
            if classify_row(r) not in {'strong_anchor', 'partial_time'}:
                continue
            s, e = get_row_time_bounds(r)
            if s is not None and (anchor_start is None or s < anchor_start):
                anchor_start = s
            if e is not None and (anchor_end is None or e > anchor_end):
                anchor_end = e
        if anchor_start and anchor_end:
            anchor_positions.append((min_row, anchor_start, anchor_end))
    anchor_positions.sort(key=lambda x: x[0])
    return anchor_positions


def _assign_unanchored_groups(
    unanchored_groups: list[tuple[datetime, datetime, list[tuple[str, list[MetadataRow], timedelta]]]],
) -> dict[str, tuple[datetime, datetime]]:
    """Distribute unanchored cluster groups evenly within their time windows."""
    result: dict[str, tuple[datetime, datetime]] = {}
    for group_start, group_end, clusters in unanchored_groups:
        available_duration = group_end - group_start
        total_spread = sum((spread for _, _, spread in clusters), timedelta())
        num_clusters = len(clusters)
        if available_duration >= total_spread and num_clusters > 0:
            gap = (available_duration - total_spread) / (num_clusters + 1)
            current_time = group_start + gap
            for name, _, spread in clusters:
                result[name] = (current_time, current_time + spread)
                current_time += spread + gap
        else:
            fraction = available_duration / total_spread if total_spread > timedelta() else 1
            current_time = group_start
            for name, _, spread in clusters:
                actual_spread = spread * fraction
                result[name] = (current_time, current_time + actual_spread)
                current_time += actual_spread
    return result


def _distribute_clusters_in_scene(
    clusters_info: list[tuple[str, list[MetadataRow], bool, timedelta]],
    scene_start: datetime,
    scene_end: datetime,
) -> dict[str, tuple[datetime, datetime]]:
    """Distribute clusters within a scene based on row sequence."""
    sorted_clusters = sorted(clusters_info, key=lambda x: min(r.row_num for r in x[1]))
    anchor_positions = _build_anchor_positions(sorted_clusters)

    unanchored_groups: list[tuple[datetime, datetime, list]] = []
    current_group: list[tuple[str, list[MetadataRow], timedelta]] = []
    current_start = scene_start

    for name, rows, has_anchors, spread in sorted_clusters:
        min_row = min(r.row_num for r in rows)
        if has_anchors and not name.startswith('__SINGLETON_'):
            if current_group:
                anchor_time = next(
                    (astart for ar, astart, _ in anchor_positions if ar == min_row),
                    scene_end,
                )
                unanchored_groups.append((current_start, anchor_time, current_group))
                current_group = []
            for ar, _, aend in anchor_positions:
                if ar == min_row:
                    current_start = aend
                    break
        else:
            current_group.append((name, rows, spread))

    if current_group:
        unanchored_groups.append((current_start, scene_end, current_group))

    return _assign_unanchored_groups(unanchored_groups)


def _handle_partial_time_cluster(
    name: str,
    rows: list[MetadataRow],
    spread: timedelta,
) -> tuple[str, tuple[datetime, datetime]] | None:
    """Compute bounds for a cluster with partial time anchors."""
    specific_anchors: list[tuple[datetime, datetime]] = []
    broad_anchors: list[tuple[datetime, datetime]] = []

    for r in rows:
        row_status = classify_row(r)
        if row_status in {'strong_anchor', 'partial_time'}:
            s, e = get_row_time_bounds(r)
            if s and e:
                specific_anchors.append((s, e))
        elif row_status == 'date_only':
            broad_anchors.append((
                datetime(r.year, r.month, r.day, 0, 0, 0),  # ty: ignore[invalid-argument-type]
                datetime(r.year, r.month, r.day, 23, 59, 59),  # ty: ignore[invalid-argument-type]
            ))

    cluster_anchors = specific_anchors if specific_anchors else broad_anchors
    if not cluster_anchors:
        return None

    allowed_start, allowed_end = span_of(cluster_anchors)
    available = allowed_end - allowed_start
    if spread > available:
        return name, (allowed_start, allowed_end)

    min_row = min(r.row_num for r in rows)
    hash_val = (min_row * 2654435761) % (2**32)
    fraction = hash_val / (2**32)
    offset = fraction * (available - spread)
    cluster_start = allowed_start + offset
    return name, (cluster_start, cluster_start + spread)


def _date_only_available_window(
    row: MetadataRow, scene_start: datetime, scene_end: datetime
) -> tuple[datetime, datetime]:
    """Return the available (start, end) window for a date_only singleton."""
    year, month, day = cast(int, row.year), cast(int, row.month), cast(int, row.day)
    day_start = datetime(year, month, day, 0, 0, 0)
    day_end = datetime(year, month, day, 23, 59, 59)
    if scene_end >= day_start and scene_start <= day_end:
        return max(day_start, scene_start), min(day_end, scene_end)
    return day_start, day_end


def _handle_constrained_singleton(
    name: str,
    rows: list[MetadataRow],
    status: str,
    scene_start: datetime,
    scene_end: datetime,
) -> tuple[str, tuple[datetime, datetime]]:
    """Compute bounds for a singleton cluster with date/partial-date constraints."""
    row = rows[0]
    spread = _SINGLETON_SPREAD

    if status == 'date_only':
        available_start, available_end = _date_only_available_window(row, scene_start, scene_end)
    elif status in {'year_month', 'year_only'}:
        start, end = get_partial_date_bounds(row)
        if start and end and scene_end >= start and scene_start <= end:
            available_start = max(start, scene_start)
            available_end = min(end, scene_end)
        else:
            available_start = start if start else scene_start
            available_end = end if end else scene_end
    else:
        available_start, available_end = scene_start, scene_end

    available_duration = available_end - available_start
    if available_duration >= spread:
        row_num = row.row_num if row.row_num is not None else 0
        hash_val = (row_num * 2654435761) % (2**32)
        fraction = hash_val / (2**32)
        offset = fraction * (available_duration - spread)
        singleton_start = available_start + offset
        singleton_end = singleton_start + spread
    else:
        singleton_start, singleton_end = available_start, available_end

    return name, (singleton_start, singleton_end)


def distribute_clusters_with_constraints(
    clusters_info: list[tuple[str, list[MetadataRow], bool, timedelta, str | None]],
    scene_start: datetime,
    scene_end: datetime,
) -> dict[str, tuple[datetime, datetime]]:
    """Distribute all clusters respecting constraints."""
    result: dict[str, tuple[datetime, datetime]] = {}
    sorted_clusters = sorted(clusters_info, key=lambda x: min(r.row_num for r in x[1]))

    scene_distributed: list[tuple[str, list[MetadataRow], bool, timedelta]] = []
    constrained_singletons: list[tuple[str, list[MetadataRow], str]] = []

    for name, rows, has_anchors, spread, status in sorted_clusters:
        if status is None or status in {'no_date', 'invalid'}:
            scene_distributed.append((name, rows, has_anchors if status is None else False, spread))
        elif status != 'cluster_partial_time':
            constrained_singletons.append((name, rows, status))

    if scene_distributed:
        result.update(_distribute_clusters_in_scene(scene_distributed, scene_start, scene_end))

    for name, rows, _, spread, status in sorted_clusters:
        if status != 'cluster_partial_time':
            continue
        entry = _handle_partial_time_cluster(name, rows, spread)
        if entry:
            result[entry[0]] = entry[1]

    for name, rows, status in constrained_singletons:
        entry = _handle_constrained_singleton(name, rows, status, scene_start, scene_end)
        result[entry[0]] = entry[1]

    return result
