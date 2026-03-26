from datetime import datetime
from typing import cast

from evergrain.core.metadata.boundary_inference.row_classification import (
    classify_row,
    get_partial_date_bounds,
    get_row_time_bounds,
)
from evergrain.core.models.metadata import MetadataRow
from evergrain.utils.validators import is_valid_date

_DEFAULT_EVENT_START = datetime(2000, 1, 1, 0, 0, 0)
_DEFAULT_EVENT_END = datetime(2000, 12, 31, 23, 59, 59)


def _collect_time_anchors(
    rows: list[MetadataRow],
) -> list[tuple[datetime, datetime]]:
    """Collect (start, end) pairs from strong/partial-time rows."""
    anchors = []
    for r in rows:
        if classify_row(r) in {'strong_anchor', 'partial_time'}:
            start, end = get_row_time_bounds(r)
            if start and end:
                anchors.append((start, end))
    return anchors


def span_of(pairs: list[tuple[datetime, datetime]]) -> tuple[datetime, datetime]:
    """Return the min start and max end across a list of (start, end) pairs."""
    return min(p[0] for p in pairs), max(p[1] for p in pairs)


def _get_event_bounds_from_anchors(
    event_rows: list[MetadataRow],
) -> tuple[datetime, datetime] | None:
    """Try to derive event bounds from strong/partial time anchors."""
    anchors = _collect_time_anchors(event_rows)
    if anchors:
        return span_of(anchors)
    return None


def _get_event_bounds_from_dates(
    event_rows: list[MetadataRow],
) -> tuple[datetime, datetime] | None:
    """Try to derive event bounds from date-only / partial-date rows."""
    pairs = []
    for r in event_rows:
        status = classify_row(r)
        if status == 'date_only':
            year, month, day = cast(int, r.year), cast(int, r.month), cast(int, r.day)
            pairs.append((
                datetime(year, month, day, 0, 0, 0),
                datetime(year, month, day, 23, 59, 59),
            ))
        elif status in {'year_month', 'year_only'}:
            start, end = get_partial_date_bounds(r)
            if start and end:
                pairs.append((start, end))
    if pairs:
        return span_of(pairs)
    return None


def get_event_bounds(event_rows: list[MetadataRow]) -> tuple[datetime, datetime]:
    """Calculate event start/end from available time anchors."""
    result = _get_event_bounds_from_anchors(event_rows)
    if result:
        return result

    result = _get_event_bounds_from_dates(event_rows)
    if result:
        return result

    years = [r.year for r in event_rows if r.year]
    if years:
        year = min(years)
        return datetime(year, 1, 1, 0, 0, 0), datetime(year, 12, 31, 23, 59, 59)

    return _DEFAULT_EVENT_START, _DEFAULT_EVENT_END


def get_scene_bounds(
    scene_rows: list[MetadataRow], event_start: datetime, event_end: datetime
) -> tuple[datetime, datetime]:
    """Calculate scene bounds using anchors or fallback to event bounds."""
    anchors = _collect_time_anchors(scene_rows)
    if anchors:
        return span_of(anchors)

    date_rows = [r for r in scene_rows if is_valid_date(r.year, r.month, r.day)]
    if date_rows:
        pairs = [
            (
                datetime(r.year, r.month, r.day, 0, 0, 0),  # ty: ignore[invalid-argument-type]
                datetime(r.year, r.month, r.day, 23, 59, 59),  # ty: ignore[invalid-argument-type]
            )
            for r in date_rows
        ]
        return span_of(pairs)

    return event_start, event_end
