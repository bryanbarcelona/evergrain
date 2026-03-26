from pathlib import Path

from evergrain.core.metadata.assignment import assign_datetime
from evergrain.core.metadata.boundary_inference import infer_temporal_bounds
from evergrain.core.metadata.csv_loader import load_metadata_csv
from evergrain.core.metadata.normalization import normalize_metadata
from evergrain.core.models.metadata import MetadataRow


class Metadata:
    """
    Class to handle metadata operations.
    """

    def __init__(self, csv_path: str | Path) -> None:
        self.metadata_rows: list[MetadataRow] = load_metadata_csv(Path(csv_path))
        self.metadata_rows: list[MetadataRow] = normalize_metadata(self.metadata_rows)
        self.metadata_rows: list[MetadataRow] = infer_temporal_bounds(self.metadata_rows)
        self.metadata_rows: list[MetadataRow] = assign_datetime(self.metadata_rows)

    def get_metadata(self) -> list[MetadataRow]:
        return self.metadata_rows

    @property
    def row_count(self) -> int:
        return len(self.metadata_rows)
