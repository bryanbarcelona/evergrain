from collections import defaultdict
from datetime import datetime, timedelta
from typing import cast

from evergrain.core.metadata.boundary_inference.cluster_distribution import (
    distribute_clusters_with_constraints,
    get_cluster_bounds,
)
from evergrain.core.metadata.boundary_inference.event_scene_bounds import get_event_bounds, get_scene_bounds
from evergrain.core.metadata.boundary_inference.row_classification import (
    classify_row,
    get_partial_date_bounds,
    get_row_time_bounds,
)
from evergrain.core.models.metadata import MetadataRow

_TIGHT_CLUSTER_SPREAD = timedelta(seconds=30)
_LOOSE_CLUSTER_SPREAD = timedelta(minutes=4)
_SINGLETON_SPREAD = timedelta(minutes=4)
_DEFAULT_EVENT_START = datetime(2000, 1, 1, 0, 0, 0)
_DEFAULT_EVENT_END = datetime(2000, 12, 31, 23, 59, 59)


def infer_temporal_bounds(rows: list[MetadataRow]) -> list[MetadataRow]:
    """Infer temporal bounds for each metadata row.

    This function processes a list of MetadataRow objects and infers the
    lower and upper temporal bounds for each row based on the hierarchical
    structure of Event, Scene, and Cluster. It uses available date/time
    components to compute these bounds, applying specific rules for clusters.

    Args:
        rows (List[MetadataRow]): List of metadata rows to process.
    Returns:
        List[MetadataRow]: The input list with updated lower_bound and upper_bound.
    """
    return _compute_bounds_for_all_rows(rows)


def _enforce_min_cluster_spread(rows: list[MetadataRow]) -> None:
    """Expand cluster bounds ONLY if current span is too small for unique assignment."""
    groups: dict[str, list[MetadataRow]] = defaultdict(list)
    for row in rows:
        if row.unique_group_id is not None:
            groups[row.unique_group_id].append(row)

    for group in groups.values():
        n = len(group)
        if n <= 1:
            continue

        starts = [r.lower_bound for r in group if r.lower_bound is not None]
        ends = [r.upper_bound for r in group if r.upper_bound is not None]
        if not starts or not ends:
            continue

        start = min(starts)
        end = max(ends)
        current_span = end - start
        cluster_name = group[0].Cluster or ''
        required_span = timedelta(seconds=5 * n) if 'tight' in cluster_name.lower() else timedelta(seconds=60 * n)

        if current_span >= required_span:
            continue

        center = start + current_span / 2
        half = required_span / 2
        new_start = center - half
        new_end = center + half
        for row in group:
            row.lower_bound = new_start
            row.upper_bound = new_end


def _resolve_no_scene_bounds(
    clusters: dict[str, list[MetadataRow]],
    scene_bounds_data: list[tuple[str, datetime, datetime, int, int]],
    event_start: datetime,
    event_end: datetime,
) -> tuple[datetime, datetime]:
    """Resolve temporal bounds for rows that belong to no named scene."""
    scene_rows_nums = [r.row_num for cluster_rows in clusters.values() for r in cluster_rows]
    min_row = min(scene_rows_nums)
    max_row = max(scene_rows_nums)
    before = [s for s in scene_bounds_data if s[4] < min_row]
    after = [s for s in scene_bounds_data if s[3] > max_row]
    if before and after:
        return before[-1][2], after[0][1]
    if before:
        return before[-1][2], event_end
    if after:
        return event_start, after[0][1]
    return event_start, event_end


def _build_distributable_clusters(
    clusters: dict[str, list[MetadataRow]],
) -> list[tuple[str, list[MetadataRow], bool, timedelta, str | None]]:
    """Build the cluster descriptor list consumed by distribute_clusters_with_constraints."""
    distributable: list[tuple[str, list[MetadataRow], bool, timedelta, str | None]] = []
    for cluster_name, cluster_rows in clusters.items():
        is_singleton = cluster_name.startswith('__SINGLETON_')
        if is_singleton:
            status = classify_row(cluster_rows[0])
            if status in {'no_date', 'invalid', 'date_only', 'year_month', 'year_only'}:
                distributable.append((cluster_name, cluster_rows, False, _TIGHT_CLUSTER_SPREAD, status))
        else:
            spread = _TIGHT_CLUSTER_SPREAD if 'tight' in cluster_name.lower() else _LOOSE_CLUSTER_SPREAD
            has_specific_time = any(classify_row(r) in {'strong_anchor', 'partial_time'} for r in cluster_rows)
            distributable.append((
                cluster_name,
                cluster_rows,
                False,
                spread,
                'cluster_partial_time' if has_specific_time else None,
            ))
    return distributable


def _assign_singleton_bounds(
    row: MetadataRow,
    cluster_name: str,
    cluster_start: datetime,
    cluster_end: datetime,
    *,
    scene_start: datetime,
    scene_end: datetime,
    distributed_bounds: dict[str, tuple[datetime, datetime]],
) -> None:
    """Set lower/upper bound on a singleton row based on its classification."""
    status = classify_row(row)
    if status == 'strong_anchor':
        year, month, day = cast(int, row.year), cast(int, row.month), cast(int, row.day)
        hour, minute, second = cast(int, row.hour), cast(int, row.minute), cast(int, row.second)
        dt = datetime(year, month, day, hour, minute, second)
        row.lower_bound = row.upper_bound = dt
    elif status == 'partial_time':
        start, end = get_row_time_bounds(row)
        row.lower_bound = max(start, scene_start) if start else None
        row.upper_bound = min(end, scene_end) if end else None
    elif status == 'date_only':
        if cluster_name in distributed_bounds:
            row.lower_bound, row.upper_bound = cluster_start, cluster_end
        else:
            year, month, day = cast(int, row.year), cast(int, row.month), cast(int, row.day)
            day_start = datetime(year, month, day, 0, 0, 0)
            day_end = datetime(year, month, day, 23, 59, 59)
            row.lower_bound = max(day_start, scene_start)
            row.upper_bound = min(day_end, scene_end)
    elif status in {'year_month', 'year_only'}:
        if cluster_name in distributed_bounds:
            row.lower_bound, row.upper_bound = cluster_start, cluster_end
        else:
            partial_start, partial_end = get_partial_date_bounds(row)
            if partial_start and partial_end:
                row.lower_bound = max(partial_start, scene_start)
                row.upper_bound = min(partial_end, scene_end)
    elif status == 'invalid':
        row.lower_bound = row.upper_bound = None
    else:  # no_date
        row.lower_bound, row.upper_bound = cluster_start, cluster_end


def _assign_cluster_bounds_in_scene(
    clusters: dict[str, list[MetadataRow]],
    distributed_bounds: dict[str, tuple[datetime, datetime]],
    scene_start: datetime,
    scene_end: datetime,
) -> None:
    """Write lower/upper bounds onto every row in the scene's clusters."""
    for cluster_name, cluster_rows in clusters.items():
        is_singleton = cluster_name.startswith('__SINGLETON_')
        if cluster_name in distributed_bounds:
            cluster_start, cluster_end = distributed_bounds[cluster_name]
        else:
            cluster_start, cluster_end = get_cluster_bounds(cluster_rows, scene_start, scene_end)

        if not is_singleton:
            for row in cluster_rows:
                row.lower_bound = cluster_start
                row.upper_bound = cluster_end
        else:
            for row in cluster_rows:
                _assign_singleton_bounds(
                    row,
                    cluster_name,
                    cluster_start,
                    cluster_end,
                    scene_start=scene_start,
                    scene_end=scene_end,
                    distributed_bounds=distributed_bounds,
                )


def _compute_bounds_for_all_rows(rows: list[MetadataRow]) -> list[MetadataRow]:
    """Compute temporal inference bounds for each row."""
    events: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        event_key = row.Event if row.Event else f'__NO_EVENT_{row.row_num}__'
        scene_key = row.Scene if row.Scene else '__NO_SCENE__'
        cluster_key = row.Cluster if row.Cluster else f'__SINGLETON_{row.row_num}__'
        events[event_key][scene_key][cluster_key].append(row)

    for _, scenes in events.items():
        event_rows = [
            r for scene_clusters in scenes.values() for cluster_rows in scene_clusters.values() for r in cluster_rows
        ]
        event_start, event_end = get_event_bounds(event_rows)

        scene_bounds_data: list[tuple[str, datetime, datetime, int, int]] = []
        for scene_name, clusters in scenes.items():
            if scene_name == '__NO_SCENE__':
                continue
            scene_rows = [r for cluster_rows in clusters.values() for r in cluster_rows]
            scene_start, scene_end = get_scene_bounds(scene_rows, event_start, event_end)
            min_row = min(r.row_num for r in scene_rows)
            max_row = max(r.row_num for r in scene_rows)
            scene_bounds_data.append((scene_name, scene_start, scene_end, min_row, max_row))

        for scene_name, clusters in scenes.items():
            if scene_name == '__NO_SCENE__':
                scene_start, scene_end = _resolve_no_scene_bounds(clusters, scene_bounds_data, event_start, event_end)
            else:
                scene_rows = [r for cluster_rows in clusters.values() for r in cluster_rows]
                scene_start, scene_end = get_scene_bounds(scene_rows, event_start, event_end)

            distributable_clusters = _build_distributable_clusters(clusters)
            distributed_bounds = distribute_clusters_with_constraints(distributable_clusters, scene_start, scene_end)
            _assign_cluster_bounds_in_scene(clusters, distributed_bounds, scene_start, scene_end)

    _enforce_min_cluster_spread(rows)
    return rows
