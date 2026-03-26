import csv
from pathlib import Path

from evergrain.core.models.metadata import MetadataRow
from evergrain.utils.validators import is_valid_date


def load_metadata_csv(csv_path: Path) -> list[MetadataRow]:
    """Load metadata from CSV file into a list of MetadataRow objects."""
    if not csv_path.exists():
        raise FileNotFoundError(f'CSV file not found: {csv_path}')

    rows: list[MetadataRow] = []
    with Path(csv_path).open(newline='', encoding='utf-8') as f:
        sample = f.read(8192)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=',;')
            delimiter = dialect.delimiter
            print(f'Detected delimiter: {delimiter!r}')
        except csv.Error:
            delimiter = ';' if sample.count(';') > sample.count(',') else ','
        reader = csv.DictReader(f, delimiter=delimiter)
        for rn, raw in enumerate(reader, start=2):
            # Parse and clamp time components
            year = _to_int_or_none(raw.get('YY', ''))
            month = _to_int_or_none(raw.get('MM', ''))
            day = _to_int_or_none(raw.get('DD', ''))
            hour = _clamp_time_component(_to_int_or_none(raw.get('HH', '')), 0, 23)
            minute = _clamp_time_component(_to_int_or_none(raw.get('MN', '')), 0, 59)
            second = _clamp_time_component(_to_int_or_none(raw.get('SS', '')), 0, 59)

            # Validate day against month/year; invalidate if impossible
            if not is_valid_date(year, month, day):
                day = None

            rows.append(
                MetadataRow(
                    Event=(raw.get('Event') or '').strip() or None,
                    Scene=(raw.get('Scene') or '').strip() or None,
                    Location=(raw.get('Location') or '').strip() or None,
                    Tags=(raw.get('Tags') or '').strip() or None,
                    Cluster=(raw.get('Cluster') or '').strip() or None,
                    year=year,
                    month=month,
                    day=day,
                    hour=hour,
                    minute=minute,
                    second=second,
                    raw_row=raw,
                    row_num=rn,
                )
            )
    return rows


def _to_int_or_none(value: str) -> int | None:
    """Convert string to int or return None if empty/invalid."""
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return int(stripped)
    except ValueError:
        return None


def _clamp_time_component(value: int | None, min_val: int, max_val: int) -> int | None:
    """Clamp time component to valid range if not None."""
    if value is None:
        return None
    return max(min_val, min(max_val, value))
