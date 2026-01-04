from datetime import datetime
from typing import Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MetadataRow:
    """Represents a single row of metadata with inferred temporal bounds."""

    Event: Optional[str] = None
    event_id: Optional[int] = None
    Scene: Optional[str] = None
    scene_id: Optional[int] = None
    Location: Optional[str] = None
    Tags: Optional[str] = None
    Cluster: Optional[str] = None
    cluster_id: Optional[int] = None
    unique_group_id: Optional[str] = None
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None
    hour: Optional[int] = None
    minute: Optional[int] = None
    second: Optional[int] = None
    upper_bound: Optional[datetime] = None
    lower_bound: Optional[datetime] = None
    photo_datetime: Optional[datetime] = None
    raw_row: Optional[dict] = None
    row_num: Optional[int] = None
    photo_filepath: Optional[Path] = None
