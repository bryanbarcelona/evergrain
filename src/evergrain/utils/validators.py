from calendar import monthrange


def is_valid_date(year: int | None, month: int | None, day: int | None) -> bool:
    """Check if year/month/day form a valid calendar date (leap-year aware)."""

    if year is None or month is None or day is None:
        return False

    if not (1 <= month <= 12):
        return False

    try:
        max_day = monthrange(year, month)[1]
    except (ValueError, TypeError):
        return False
    else:
        return 1 <= day <= max_day
