from evergrain.core.exceptions.base import EverGrainError


class EdgeReachedException(EverGrainError, StopIteration):
    """Exception raised when an edge is reached during segmentation."""
