from typing import Optional
from calendar import monthrange


def _is_valid_date(
    year: Optional[int], month: Optional[int], day: Optional[int]
) -> bool:
    """Check if year/month/day form a valid calendar date (leap-year aware)."""
    if None in (year, month, day):
        return False
    if not (1 <= month <= 12):
        return False
    try:
        max_day = monthrange(year, month)[1]
        return 1 <= day <= max_day
    except (ValueError, TypeError):
        return False

