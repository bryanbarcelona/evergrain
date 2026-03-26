import random
from collections import defaultdict
from datetime import datetime, timedelta

from evergrain.core.models.metadata import MetadataRow


def assign_datetime(metadata_rows: list[MetadataRow]) -> list[MetadataRow]:
    groups = _group_by_unique_id(metadata_rows)

    if not _check_group_consistency(groups):
        return []

    max_attempts = 100

    for _ in range(max_attempts):
        assigned_rows = _assign_and_normalize(groups)
        duplicates = _find_duplicates(assigned_rows)

        if not duplicates:
            assigned_rows.sort(key=lambda r: r.row_num)
            return assigned_rows

    # Final attempt for error reporting
    return _handle_final_attempt(groups, max_attempts)


def _group_by_unique_id(metadata_rows: list[MetadataRow]) -> dict[str, list[MetadataRow]]:
    """Group metadata rows by unique_group_id."""
    groups = defaultdict(list)
    for row in metadata_rows:
        if row.unique_group_id:
            groups[row.unique_group_id].append(row)
    return groups


def _assign_and_normalize(groups: dict[str, list[MetadataRow]]) -> list[MetadataRow]:
    """Assign dates per group and normalize all photo_datetimes."""
    assigned = []
    for rows in groups.values():
        _assign_date(rows)
        assigned.extend(rows)
    _normalize_photo_datetimes(assigned)
    return assigned


def _find_duplicates(rows: list[MetadataRow]) -> dict[datetime, list[int]]:
    """Return datetime -> list of row_nums for duplicated datetimes."""
    dt_to_rows = defaultdict(list)
    for row in rows:
        if row.photo_datetime is not None:
            dt_to_rows[row.photo_datetime].append(row.row_num)

    return {dt: nums for dt, nums in dt_to_rows.items() if len(nums) > 1}


def _handle_final_attempt(groups: dict[str, list[MetadataRow]], max_attempts: int) -> list[MetadataRow]:
    """Handle the final attempt after max attempts, raising appropriate errors."""
    assigned_rows = _assign_and_normalize(groups)
    duplicates = _find_duplicates(assigned_rows)

    if not duplicates:
        raise RuntimeError('Unexpected: max attempts reached but no duplicates found on final check.')

    conflict_details = [f'  {dt} → Rows: {sorted(rows)}' for dt, rows in duplicates.items()]

    raise RuntimeError(
        f'Failed to assign unique photo_datetimes after {max_attempts} attempts.\n'
        f'Final conflicting timestamps (at second resolution):\n' + '\n'.join(conflict_details)
    )


# ----------------------------
# Private Helper Functions
# ----------------------------


def _check_group_consistency(groups: dict[str, list[MetadataRow]]) -> bool:
    for _, rows in groups.items():
        upper_bounds = [row.upper_bound for row in rows]
        lower_bounds = [row.lower_bound for row in rows]

        upper_same = all(bound == upper_bounds[0] for bound in upper_bounds)
        lower_same = all(bound == lower_bounds[0] for bound in lower_bounds)

        if not (upper_same and lower_same):  # ← both must be consistent
            return False
    return True


def _assign_date(rows: list[MetadataRow]) -> None:
    """Assign random datetime values to each row in the group."""
    interval_count = len(rows)
    if interval_count == 0:
        return

    start_time = min(row.lower_bound for row in rows)
    end_time = max(row.upper_bound for row in rows)
    delta = end_time - start_time
    interval_duration = delta / interval_count

    for row in rows:
        section_endtime = start_time + interval_duration

        row.photo_datetime = _random_datetime_beta(start_time, section_endtime)

        start_time = section_endtime


def _random_datetime_beta(start_datetime: datetime, end_datetime: datetime, alpha: int = 5, beta: int = 5) -> datetime:
    """Generate a random datetime between start and end using beta distribution."""
    delta_seconds = (end_datetime - start_datetime).total_seconds()
    seconds = random.betavariate(alpha, beta) * delta_seconds
    return start_datetime + timedelta(seconds=seconds)


def _normalize_photo_datetimes(rows: list[MetadataRow]) -> None:
    """
    Remove microseconds from the `photo_datetime` field of each MetadataRow.
    Modifies the list in place.
    """
    for row in rows:
        if row.photo_datetime is not None:
            row.photo_datetime = row.photo_datetime.replace(microsecond=0)
