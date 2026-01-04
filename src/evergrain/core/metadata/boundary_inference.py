from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from evergrain.core.models.metadata import MetadataRow
from evergrain.utils.validators import _is_valid_date


_TIGHT_CLUSTER_SPREAD = timedelta(seconds=30)
_LOOSE_CLUSTER_SPREAD = timedelta(minutes=4)
_SINGLETON_SPREAD = timedelta(minutes=4)
_DEFAULT_EVENT_START = datetime(2000, 1, 1, 0, 0, 0)
_DEFAULT_EVENT_END = datetime(2000, 12, 31, 23, 59, 59)


def infer_temporal_bounds(rows: List[MetadataRow]) -> List[MetadataRow]:
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


# ----------------------------
# Private Helper Functions
# ----------------------------


def _classify_row(row: MetadataRow) -> str:
    """Classify row based on available datetime components."""
    has_valid_date = _is_valid_date(row.year, row.month, row.day)
    has_hour = row.hour is not None
    has_minute = row.minute is not None
    has_second = row.second is not None

    if has_valid_date and has_hour and has_minute and has_second:
        return "strong_anchor"
    elif has_valid_date and (has_hour or has_minute or has_second):
        return "partial_time"
    elif has_valid_date:
        return "date_only"
    elif row.year is not None and row.month is not None and row.day is None:
        return "year_month"
    elif row.year is not None and row.month is None and row.day is None:
        return "year_only"
    elif any(x is not None for x in (row.year, row.month, row.day)):
        return "invalid"
    else:
        return "no_date"


def _get_row_time_bounds(
    row: MetadataRow,
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Get min/max time for a row based on provided time components."""
    if not _is_valid_date(row.year, row.month, row.day):
        return None, None
    if row.hour is not None:
        if row.minute is not None:
            if row.second is not None:
                dt = datetime(row.year, row.month, row.day, row.hour, row.minute, row.second)
                return dt, dt
            else:
                start = datetime(row.year, row.month, row.day, row.hour, row.minute, 0)
                end = datetime(row.year, row.month, row.day, row.hour, row.minute, 59)
                return start, end
        else:
            start = datetime(row.year, row.month, row.day, row.hour, 0, 0)
            end = datetime(row.year, row.month, row.day, row.hour, 59, 59)
            return start, end
    else:
        start = datetime(row.year, row.month, row.day, 0, 0, 0)
        end = datetime(row.year, row.month, row.day, 23, 59, 59)
        return start, end


def _get_partial_date_bounds(
    row: MetadataRow,
) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Get bounds for year+month or year-only rows."""
    if row.year is not None and row.month is not None:
        try:
            start = datetime(row.year, row.month, 1, 0, 0, 0)
            if row.month == 12:
                end = datetime(row.year, 12, 31, 23, 59, 59)
            else:
                next_month = datetime(row.year, row.month + 1, 1, 0, 0, 0)
                end = next_month - timedelta(seconds=1)
            return start, end
        except (ValueError, TypeError):
            return None, None
    elif row.year is not None:
        try:
            start = datetime(row.year, 1, 1, 0, 0, 0)
            end = datetime(row.year, 12, 31, 23, 59, 59)
            return start, end
        except (ValueError, TypeError):
            return None, None
    return None, None


def _get_event_bounds(event_rows: List[MetadataRow]) -> Tuple[datetime, datetime]:
    """Calculate event start/end from available time anchors."""
    anchors = []
    for r in event_rows:
        status = _classify_row(r)
        if status in ["strong_anchor", "partial_time"]:
            start, end = _get_row_time_bounds(r)
            if start and end:
                anchors.append((start, end))
    if anchors:
        all_starts = [a[0] for a in anchors]
        all_ends = [a[1] for a in anchors]
        return min(all_starts), max(all_ends)

    partial_date_rows = []
    for r in event_rows:
        status = _classify_row(r)
        if status in ["date_only", "year_month", "year_only"]:
            if status == "date_only":
                start = datetime(r.year, r.month, r.day, 0, 0, 0)
                end = datetime(r.year, r.month, r.day, 23, 59, 59)
            else:
                start, end = _get_partial_date_bounds(r)
            if start and end:
                partial_date_rows.append((start, end))
    if partial_date_rows:
        all_starts = [p[0] for p in partial_date_rows]
        all_ends = [p[1] for p in partial_date_rows]
        return min(all_starts), max(all_ends)

    date_rows = [r for r in event_rows if _is_valid_date(r.year, r.month, r.day)]
    if date_rows:
        dates = [datetime(r.year, r.month, r.day, 0, 0, 0) for r in date_rows]
        end_dates = [datetime(r.year, r.month, r.day, 23, 59, 59) for r in date_rows]
        return min(dates), max(end_dates)

    years = [r.year for r in event_rows if r.year]
    if years:
        year = min(years)
        return datetime(year, 1, 1, 0, 0, 0), datetime(year, 12, 31, 23, 59, 59)

    return _DEFAULT_EVENT_START, _DEFAULT_EVENT_END


def _get_scene_bounds(
    scene_rows: List[MetadataRow], event_start: datetime, event_end: datetime
) -> Tuple[datetime, datetime]:
    """Calculate scene bounds using anchors or fallback to event bounds."""
    anchors = []
    for r in scene_rows:
        status = _classify_row(r)
        if status in ["strong_anchor", "partial_time"]:
            start, end = _get_row_time_bounds(r)
            if start and end:
                anchors.append((start, end))
    if anchors:
        all_starts = [a[0] for a in anchors]
        all_ends = [a[1] for a in anchors]
        return min(all_starts), max(all_ends)

    date_rows = [r for r in scene_rows if _is_valid_date(r.year, r.month, r.day)]
    if date_rows:
        dates = [datetime(r.year, r.month, r.day, 0, 0, 0) for r in date_rows]
        end_dates = [datetime(r.year, r.month, r.day, 23, 59, 59) for r in date_rows]
        return min(dates), max(end_dates)

    return event_start, event_end


def _get_cluster_bounds(
    cluster_rows: List[MetadataRow], scene_start: datetime, scene_end: datetime
) -> Tuple[datetime, datetime]:
    """Calculate cluster bounds with spread rules."""
    specific_anchors = []
    broad_anchors = []
    for r in cluster_rows:
        status = _classify_row(r)
        if status in ["strong_anchor", "partial_time"]:
            start, end = _get_row_time_bounds(r)
            if start and end:
                specific_anchors.append((start, end))
        elif status == "date_only":
            start = datetime(r.year, r.month, r.day, 0, 0, 0)
            end = datetime(r.year, r.month, r.day, 23, 59, 59)
            broad_anchors.append((start, end))
        elif status in ["year_month", "year_only"]:
            start, end = _get_partial_date_bounds(r)
            if start and end:
                broad_anchors.append((start, end))

    anchors = specific_anchors if specific_anchors else broad_anchors
    cluster_name = cluster_rows[0].Cluster if cluster_rows else None

    if anchors:
        all_starts = [a[0] for a in anchors]
        all_ends = [a[1] for a in anchors]
        allowed_start = min(all_starts)
        allowed_end = max(all_ends)
        available = allowed_end - allowed_start

        if cluster_name and "tight" in cluster_name.lower():
            spread = _TIGHT_CLUSTER_SPREAD
        elif cluster_name and "loose" in cluster_name.lower():
            spread = _LOOSE_CLUSTER_SPREAD
        else:
            spread = allowed_end - allowed_start

        if spread > available:
            return allowed_start, allowed_end
        else:
            center = allowed_start + available / 2
            cluster_start = center - spread / 2
            cluster_end = center + spread / 2
            return cluster_start, cluster_end

    else:
        if cluster_name and "tight" in cluster_name.lower():
            default_spread = _TIGHT_CLUSTER_SPREAD
        else:
            default_spread = _SINGLETON_SPREAD

        available_span = scene_end - scene_start
        actual_spread = min(default_spread, available_span)
        scene_center = scene_start + available_span / 2
        cluster_start = max(scene_start, scene_center - actual_spread / 2)
        cluster_end = min(scene_end, scene_center + actual_spread / 2)
        return cluster_start, cluster_end


def _distribute_clusters_in_scene(
    clusters_info: List[Tuple[str, List[MetadataRow], bool, timedelta]],
    scene_start: datetime,
    scene_end: datetime,
) -> dict:
    """Distribute clusters within a scene based on row sequence."""
    from datetime import timedelta  # ensure in scope

    sorted_clusters = sorted(clusters_info, key=lambda x: min(r.row_num for r in x[1]))
    anchor_positions = []
    for name, rows, has_anchors, _ in sorted_clusters:
        if has_anchors and not name.startswith("__SINGLETON_"):
            min_row = min(r.row_num for r in rows)
            anchor_start = anchor_end = None
            for r in rows:
                status = _classify_row(r)
                if status in ["strong_anchor", "partial_time"]:
                    s, e = _get_row_time_bounds(r)
                    if anchor_start is None or s < anchor_start:
                        anchor_start = s
                    if anchor_end is None or e > anchor_end:
                        anchor_end = e
            if anchor_start and anchor_end:
                anchor_positions.append((min_row, anchor_start, anchor_end))
    anchor_positions.sort(key=lambda x: x[0])

    result = {}
    unanchored_groups = []
    current_group = []
    current_start = scene_start

    for name, rows, has_anchors, spread in sorted_clusters:
        min_row = min(r.row_num for r in rows)
        if has_anchors and not name.startswith("__SINGLETON_"):
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

    for group_start, group_end, clusters in unanchored_groups:
        available_duration = group_end - group_start
        total_spread = sum((spread for _, _, spread in clusters), timedelta())  # ✅ FIXED HERE
        num_clusters = len(clusters)
        if available_duration >= total_spread and num_clusters > 0:
            gap = (available_duration - total_spread) / (num_clusters + 1)
            current_time = group_start + gap
            for name, rows, spread in clusters:
                result[name] = (current_time, current_time + spread)
                current_time += spread + gap
        else:
            fraction = available_duration / total_spread if total_spread > timedelta() else 1
            current_time = group_start
            for name, rows, spread in clusters:
                actual_spread = spread * fraction
                result[name] = (current_time, current_time + actual_spread)
                current_time += actual_spread
    return result


def _distribute_clusters_with_constraints(
    clusters_info: List[Tuple[str, List[MetadataRow], bool, timedelta, Optional[str]]],
    scene_start: datetime,
    scene_end: datetime,
) -> dict:
    """Distribute all clusters respecting constraints."""
    result = {}
    sorted_clusters = sorted(clusters_info, key=lambda x: min(r.row_num for r in x[1]))
    scene_distributed = []
    constrained_singletons = []

    for name, rows, has_anchors, spread, status in sorted_clusters:
        if status is None or status in ["no_date", "invalid"]:
            scene_distributed.append((name, rows, has_anchors if status is None else False, spread))
        elif status != "cluster_partial_time":
            constrained_singletons.append((name, rows, status))

    if scene_distributed:
        scene_results = _distribute_clusters_in_scene(scene_distributed, scene_start, scene_end)
        result.update(scene_results)

    for name, rows, has_anchors, spread, status in sorted_clusters:
        if status == "cluster_partial_time":
            specific_anchors = []
            broad_anchors = []
            for r in rows:
                row_status = _classify_row(r)
                if row_status in ["strong_anchor", "partial_time"]:
                    s, e = _get_row_time_bounds(r)
                    if s and e:
                        specific_anchors.append((s, e))
                elif row_status == "date_only":
                    s = datetime(r.year, r.month, r.day, 0, 0, 0)
                    e = datetime(r.year, r.month, r.day, 23, 59, 59)
                    broad_anchors.append((s, e))
            cluster_anchors = specific_anchors if specific_anchors else broad_anchors
            if cluster_anchors:
                all_starts = [a[0] for a in cluster_anchors]
                all_ends = [a[1] for a in cluster_anchors]
                allowed_start = min(all_starts)
                allowed_end = max(all_ends)
                available = allowed_end - allowed_start
                if spread > available:
                    result[name] = (allowed_start, allowed_end)
                else:
                    min_row = min(r.row_num for r in rows)
                    hash_val = (min_row * 2654435761) % (2**32)
                    fraction = hash_val / (2**32)
                    offset = fraction * (available - spread)
                    cluster_start = allowed_start + offset
                    cluster_end = cluster_start + spread
                    result[name] = (cluster_start, cluster_end)

    for name, rows, status in constrained_singletons:
        row = rows[0]
        spread = _SINGLETON_SPREAD
        if status == "date_only":
            day_start = datetime(row.year, row.month, row.day, 0, 0, 0)
            day_end = datetime(row.year, row.month, row.day, 23, 59, 59)
            if scene_end >= day_start and scene_start <= day_end:
                available_start = max(day_start, scene_start)
                available_end = min(day_end, scene_end)
            else:
                available_start, available_end = day_start, day_end
        elif status in ("year_month", "year_only"):
            start, end = _get_partial_date_bounds(row)
            if start and end and scene_end >= start and scene_start <= end:
                available_start = max(start, scene_start)
                available_end = min(end, scene_end)
            else:
                available_start, available_end = (start, end) if start and end else (scene_start, scene_end)
        else:
            available_start, available_end = scene_start, scene_end

        available_duration = available_end - available_start
        if available_duration >= spread:
            hash_val = (row.row_num * 2654435761) % (2**32)
            fraction = hash_val / (2**32)
            offset = fraction * (available_duration - spread)
            singleton_start = available_start + offset
            singleton_end = singleton_start + spread
        else:
            singleton_start, singleton_end = available_start, available_end
        result[name] = (singleton_start, singleton_end)

    return result


def _enforce_min_cluster_spread(rows: List[MetadataRow]) -> None:
    """Expand cluster bounds ONLY if current span is too small for unique assignment."""
    groups = defaultdict(list)
    for row in rows:
        if row.unique_group_id is not None:
            groups[row.unique_group_id].append(row)

    for _, group in groups.items():
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

        cluster_name = group[0].Cluster or ""
        if "tight" in cluster_name.lower():
            required_span = timedelta(seconds=5 * n)
        else:
            required_span = timedelta(seconds=60 * n)

        if current_span >= required_span:
            continue

        center = start + current_span / 2
        half = required_span / 2
        new_start = center - half
        new_end = center + half

        for row in group:
            row.lower_bound = new_start
            row.upper_bound = new_end


def _compute_bounds_for_all_rows(rows: List[MetadataRow]) -> List[MetadataRow]:
    """Compute temporal inference bounds for each row."""
    events = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for row in rows:
        event_key = row.Event if row.Event else f"__NO_EVENT_{row.row_num}__"
        scene_key = row.Scene if row.Scene else "__NO_SCENE__"
        cluster_key = row.Cluster if row.Cluster else f"__SINGLETON_{row.row_num}__"
        events[event_key][scene_key][cluster_key].append(row)

    for event_name, scenes in events.items():
        event_rows = [
            r for scene_clusters in scenes.values() for cluster_rows in scene_clusters.values() for r in cluster_rows
        ]
        event_start, event_end = _get_event_bounds(event_rows)
        scene_bounds_data = []
        for scene_name, clusters in scenes.items():
            if scene_name == "__NO_SCENE__":
                continue
            scene_rows = [r for cluster_rows in clusters.values() for r in cluster_rows]
            scene_start, scene_end = _get_scene_bounds(scene_rows, event_start, event_end)
            min_row = min(r.row_num for r in scene_rows)
            max_row = max(r.row_num for r in scene_rows)
            scene_bounds_data.append((scene_name, scene_start, scene_end, min_row, max_row))

        for scene_name, clusters in scenes.items():
            scene_rows = [r for cluster_rows in clusters.values() for r in cluster_rows]
            if scene_name == "__NO_SCENE__":
                scene_rows_nums = [r.row_num for cluster_rows in clusters.values() for r in cluster_rows]
                min_row = min(scene_rows_nums)
                max_row = max(scene_rows_nums)
                before = [s for s in scene_bounds_data if s[4] < min_row]
                after = [s for s in scene_bounds_data if s[3] > max_row]
                if before and after:
                    scene_start, scene_end = before[-1][2], after[0][1]
                elif before:
                    scene_start, scene_end = before[-1][2], event_end
                elif after:
                    scene_start, scene_end = event_start, after[0][1]
                else:
                    scene_start, scene_end = event_start, event_end
            else:
                scene_start, scene_end = _get_scene_bounds(scene_rows, event_start, event_end)

            distributable_clusters = []
            for cluster_name, cluster_rows in clusters.items():
                is_singleton = cluster_name.startswith("__SINGLETON_")
                if is_singleton:
                    status = _classify_row(cluster_rows[0])
                    if status in [
                        "no_date",
                        "invalid",
                        "date_only",
                        "year_month",
                        "year_only",
                    ]:
                        spread = _TIGHT_CLUSTER_SPREAD
                        distributable_clusters.append((cluster_name, cluster_rows, False, spread, status))
                else:
                    spread = _TIGHT_CLUSTER_SPREAD if "tight" in cluster_name.lower() else _LOOSE_CLUSTER_SPREAD
                    has_specific_time = any(_classify_row(r) in ["strong_anchor", "partial_time"] for r in cluster_rows)
                    distributable_clusters.append((
                        cluster_name,
                        cluster_rows,
                        False,
                        spread,
                        "cluster_partial_time" if has_specific_time else None,
                    ))

            distributed_bounds = _distribute_clusters_with_constraints(distributable_clusters, scene_start, scene_end)

            for cluster_name, cluster_rows in clusters.items():
                is_singleton = cluster_name.startswith("__SINGLETON_")
                if cluster_name in distributed_bounds:
                    cluster_start, cluster_end = distributed_bounds[cluster_name]
                else:
                    cluster_start, cluster_end = _get_cluster_bounds(cluster_rows, scene_start, scene_end)

                if not is_singleton:
                    for row in cluster_rows:
                        row.lower_bound = cluster_start
                        row.upper_bound = cluster_end
                else:
                    for row in cluster_rows:
                        status = _classify_row(row)
                        if status == "strong_anchor":
                            dt = datetime(
                                row.year,
                                row.month,
                                row.day,
                                row.hour,
                                row.minute,
                                row.second,
                            )
                            row.lower_bound = row.upper_bound = dt
                        elif status == "partial_time":
                            start, end = _get_row_time_bounds(row)
                            row.lower_bound = max(start, scene_start) if start else None
                            row.upper_bound = min(end, scene_end) if end else None
                        elif status == "date_only":
                            if cluster_name in distributed_bounds:
                                row.lower_bound, row.upper_bound = (
                                    cluster_start,
                                    cluster_end,
                                )
                            else:
                                day_start = datetime(row.year, row.month, row.day, 0, 0, 0)
                                day_end = datetime(row.year, row.month, row.day, 23, 59, 59)
                                row.lower_bound = max(day_start, scene_start)
                                row.upper_bound = min(day_end, scene_end)
                        elif status in ("year_month", "year_only"):
                            if cluster_name in distributed_bounds:
                                row.lower_bound, row.upper_bound = (
                                    cluster_start,
                                    cluster_end,
                                )
                            else:
                                partial_start, partial_end = _get_partial_date_bounds(row)
                                if partial_start and partial_end:
                                    row.lower_bound = max(partial_start, scene_start)
                                    row.upper_bound = min(partial_end, scene_end)
                        elif status == "invalid":
                            row.lower_bound = row.upper_bound = None
                        else:  # no_date
                            row.lower_bound, row.upper_bound = (
                                cluster_start,
                                cluster_end,
                            )

    _enforce_min_cluster_spread(rows)
    return rows
