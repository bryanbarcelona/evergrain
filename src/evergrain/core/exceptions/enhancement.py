from evergrain.core.exceptions.base import EverGrainError


class EnhancementError(EverGrainError):
    """Raised when image enhancement fails."""


class InvalidImageError(EnhancementError):
    """Raised when the input image cannot be opened or is in an unsupported format."""
