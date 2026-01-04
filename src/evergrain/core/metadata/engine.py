from pathlib import Path

from evergrain.core.metadata.csv_loader import load_metadata_csv
from evergrain.core.metadata.normalization import normalize_metadata
from evergrain.core.metadata.boundary_inference import infer_temporal_bounds
from evergrain.core.metadata.assignment import assign_datetime
from evergrain.core.models.metadata import MetadataRow
from typing import List


class Metadata:
    """
    Class to handle metadata operations.
    """

    def __init__(self, csv_path: str | Path) -> None:
        self.metadata_rows: List[MetadataRow] = load_metadata_csv(Path(csv_path))
        self.metadata_rows: List[MetadataRow] = normalize_metadata(self.metadata_rows)
        self.metadata_rows: List[MetadataRow] = infer_temporal_bounds(self.metadata_rows)
        self.metadata_rows: List[MetadataRow] = assign_datetime(self.metadata_rows)

    def get_metadata(self) -> List[MetadataRow]:
        return self.metadata_rows

    @property
    def row_count(self) -> int:
        return len(self.metadata_rows)
