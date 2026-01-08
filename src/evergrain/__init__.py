import importlib.metadata

try:
    __version__ = importlib.metadata.version("evergrain")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from evergrain.core.metadata.engine import Metadata
from evergrain.core.segmentation.engine import ScanBackground, PhotoSplitter
from evergrain.core.discovery.engine import Discovery
from evergrain.core.exceptions.discovery import DiscoveryError
from evergrain.core.exceptions.base import EverGrainError
from evergrain.core.enhancement.engine import EnhancementEngine
from evergrain.core.enhancement.params import EnhancementParams

__all__ = [
    "Metadata",
    "ScanBackground",
    "PhotoSplitter",
    "Discovery",
    "DiscoveryError",
    "EverGrainError",
    "EnhancementEngine",
    "EnhancementParams",
]
