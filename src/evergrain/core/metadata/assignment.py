from evergrain.core.models.metadata import MetadataRow
from typing import List, Dict
from collections import defaultdict
import random
from datetime import timedelta, datetime


def assign_datetime(metadata_rows: List[MetadataRow]) -> List[MetadataRow]:
    groups = defaultdict(list)
    for row in metadata_rows:
        if row.unique_group_id:
            groups[row.unique_group_id].append(row)

    if not _check_group_consistency(groups):
        return []

    def assign_and_normalize() -> List[MetadataRow]:
        """Assign dates per group and normalize all photo_datetimes."""
        assigned = []
        for rows in groups.values():
            _assign_date(rows)
            assigned.extend(rows)
        _normalize_photo_datetimes(assigned)
        return assigned

    def find_duplicates(rows: List[MetadataRow]) -> Dict[datetime, List[int]]:
        """Return datetime -> list of row_nums for duplicated datetimes."""
        dt_to_rows = defaultdict(list)
        for row in rows:
            if row.photo_datetime is not None:
                dt_to_rows[row.photo_datetime].append(row.row_num)

        return {dt: nums for dt, nums in dt_to_rows.items() if len(nums) > 1}

    max_attempts = 100

    for _ in range(max_attempts):
        assigned_rows = assign_and_normalize()
        duplicates = find_duplicates(assigned_rows)

        if not duplicates:
            assigned_rows.sort(key=lambda r: r.row_num)
            return assigned_rows

    # Final attempt for error reporting
    assigned_rows = assign_and_normalize()
    duplicates = find_duplicates(assigned_rows)

    if not duplicates:
        raise RuntimeError("Unexpected: max attempts reached but no duplicates found on final check.")

    conflict_details = [f"  {dt} → Rows: {sorted(rows)}" for dt, rows in duplicates.items()]

    raise RuntimeError(
        f"Failed to assign unique photo_datetimes after {max_attempts} attempts.\n"
        f"Final conflicting timestamps (at second resolution):\n" + "\n".join(conflict_details)
    )


# ----------------------------
# Private Helper Functions
# ----------------------------


def _check_group_consistency(groups: Dict[str, List[MetadataRow]]) -> bool:
    for _, rows in groups.items():
        upper_bounds = [row.upper_bound for row in rows]
        lower_bounds = [row.lower_bound for row in rows]

        upper_same = all(bound == upper_bounds[0] for bound in upper_bounds)
        lower_same = all(bound == lower_bounds[0] for bound in lower_bounds)

        if not (upper_same and lower_same):  # ← both must be consistent
            return False
    return True


def _assign_date(group: Dict[str, List[MetadataRow]]) -> None:

    interval_count = len(group)
    if interval_count == 0:
        return None

    start_time = min(row.lower_bound for row in group)
    end_time = max(row.upper_bound for row in group)
    delta = end_time - start_time
    interval_duration = delta / interval_count

    for row in group:
        section_endtime = start_time + interval_duration
        # print(f"    Row {row.row_num}: Will find datetime between {start_time} and {section_endtime}")

        row.photo_datetime = _random_datetime_beta(start_time, section_endtime)

        # print(
        #     f"    Row {row.row_num}: Assigned datetime {row.photo_datetime} from bounds {start_time} to {section_endtime}"
        # )

        start_time = section_endtime


def _random_datetime_beta(start_datetime, end_datetime, alpha=5, beta=5):
    delta_seconds = (end_datetime - start_datetime).total_seconds()
    seconds = random.betavariate(alpha, beta) * delta_seconds
    return start_datetime + timedelta(seconds=seconds)


def _normalize_photo_datetimes(rows: List[MetadataRow]) -> None:
    """
    Remove microseconds from the `photo_datetime` field of each MetadataRow.
    Modifies the list in place.
    """
    for row in rows:
        if row.photo_datetime is not None:
            row.photo_datetime = row.photo_datetime.replace(microsecond=0)
