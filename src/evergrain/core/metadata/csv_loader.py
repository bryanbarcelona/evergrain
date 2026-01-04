import csv
from pathlib import Path
from typing import List, Optional

from evergrain.core.models.metadata import MetadataRow
from evergrain.utils.validators import _is_valid_date


def load_metadata_csv(csv_path: Path) -> List[MetadataRow]:
    """Load metadata from CSV file into a list of MetadataRow objects."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    rows: List[MetadataRow] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        sample = f.read(2048)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";"])
            delimiter = dialect.delimiter
        except csv.Error:
            delimiter = ","
        reader = csv.DictReader(f, delimiter=delimiter)
        for rn, raw in enumerate(reader, start=2):
            # Parse and clamp time components
            year = _to_int_or_none(raw.get("YY", ""))
            month = _to_int_or_none(raw.get("MM", ""))
            day = _to_int_or_none(raw.get("DD", ""))
            hour = _clamp_time_component(_to_int_or_none(raw.get("HH", "")), 0, 23)
            minute = _clamp_time_component(_to_int_or_none(raw.get("MN", "")), 0, 59)
            second = _clamp_time_component(_to_int_or_none(raw.get("SS", "")), 0, 59)

            # Validate day against month/year; invalidate if impossible
            if not _is_valid_date(year, month, day):
                day = None

            rows.append(
                MetadataRow(
                    Event=(raw.get("Event") or "").strip() or None,
                    Scene=(raw.get("Scene") or "").strip() or None,
                    Location=(raw.get("Location") or "").strip() or None,
                    Tags=(raw.get("Tags") or "").strip() or None,
                    Cluster=(raw.get("Cluster") or "").strip() or None,
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


# ----------------------------
# Private Helper Functions
# ----------------------------


def _to_int_or_none(value: str) -> Optional[int]:
    """Convert string to int or return None if empty/invalid."""
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return int(stripped)
    except ValueError:
        return None


def _clamp_time_component(value: Optional[int], min_val: int, max_val: int) -> Optional[int]:
    """Clamp time component to valid range if not None."""
    if value is None:
        return None
    return max(min_val, min(max_val, value))
