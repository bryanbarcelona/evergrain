import importlib.metadata

try:
    __version__ = importlib.metadata.version("evergrain")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from evergrain.core.metadata.engine import Metadata
from evergrain.core.segmentation.engine import ScanBackground, PhotoSplitter

__all__ = ["Metadata", "ScanBackground", "PhotoSplitter"]
