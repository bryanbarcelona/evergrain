from datetime import datetime, timedelta

from evergrain.core.models.metadata import MetadataRow
from evergrain.utils.validators import is_valid_date


def classify_row(row: MetadataRow) -> str:
    """Classify row based on available datetime components."""
    has_valid_date = is_valid_date(row.year, row.month, row.day)
    has_time = row.hour is not None or row.minute is not None or row.second is not None
    has_full_time = row.hour is not None and row.minute is not None and row.second is not None

    if has_valid_date and has_full_time:
        return 'strong_anchor'
    if has_valid_date and has_time:
        return 'partial_time'
    if has_valid_date:
        return 'date_only'

    YEAR_ONLY_STATUSES = {
        (True, True, False): 'year_month',
        (True, False, False): 'year_only',
    }
    year_key = (row.year is not None, row.month is not None, row.day is not None)
    if year_key in YEAR_ONLY_STATUSES:
        return YEAR_ONLY_STATUSES[year_key]
    if any(x is not None for x in (row.year, row.month, row.day)):
        return 'invalid'
    return 'no_date'


def get_row_time_bounds(row: MetadataRow) -> tuple[datetime | None, datetime | None]:
    """Get min/max time for a row based on provided time components."""
    if not is_valid_date(row.year, row.month, row.day):
        return None, None
    # is_valid_date guarantees year/month/day are all int at this point
    year: int = row.year  # ty: ignore[invalid-assignment]
    month: int = row.month  # ty: ignore[invalid-assignment]
    day: int = row.day  # ty: ignore[invalid-assignment]
    if row.hour is not None:
        if row.minute is not None:
            if row.second is not None:
                dt = datetime(year, month, day, row.hour, row.minute, row.second)
                return dt, dt
            start = datetime(year, month, day, row.hour, row.minute, 0)
            end = datetime(year, month, day, row.hour, row.minute, 59)
            return start, end
        start = datetime(year, month, day, row.hour, 0, 0)
        end = datetime(year, month, day, row.hour, 59, 59)
        return start, end
    start = datetime(year, month, day, 0, 0, 0)
    end = datetime(year, month, day, 23, 59, 59)
    return start, end


def get_partial_date_bounds(row: MetadataRow) -> tuple[datetime | None, datetime | None]:
    """Get bounds for year+month or year-only rows."""
    try:
        if row.year is not None and row.month is not None:
            year: int = row.year
            month: int = row.month
            start = datetime(year, month, 1, 0, 0, 0)
            if month == 12:
                end = datetime(year, 12, 31, 23, 59, 59)
            else:
                next_month = datetime(year, month + 1, 1, 0, 0, 0)
                end = next_month - timedelta(seconds=1)
            return start, end
    except (ValueError, TypeError):
        pass
    try:
        if row.year is not None:
            year = row.year
            start = datetime(year, 1, 1, 0, 0, 0)
            end = datetime(year, 12, 31, 23, 59, 59)
            return start, end
    except (ValueError, TypeError):
        pass
    return None, None
