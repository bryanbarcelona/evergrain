import importlib.metadata

try:
    __version__ = importlib.metadata.version("evergrain")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from evergrain.core.metadata.engine import Metadata

__all__ = ["Metadata"]
