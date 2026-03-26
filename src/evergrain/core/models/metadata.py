from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class MetadataRow:
    """Represents a single row of metadata with inferred temporal bounds."""

    Event: str | None = None
    event_id: int | None = None
    Scene: str | None = None
    scene_id: int | None = None
    Location: str | None = None
    Tags: str | None = None
    Cluster: str | None = None
    cluster_id: int | None = None
    unique_group_id: str | None = None
    year: int | None = None
    month: int | None = None
    day: int | None = None
    hour: int | None = None
    minute: int | None = None
    second: int | None = None
    upper_bound: datetime | None = None
    lower_bound: datetime | None = None
    photo_datetime: datetime | None = None
    raw_row: dict | None = None
    row_num: int | None = None
    photo_filepath: Path | None = None
