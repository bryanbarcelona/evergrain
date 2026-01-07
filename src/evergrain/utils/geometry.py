import math
from typing import Tuple


def calculate_rotation_angle(dx: int, dy: int) -> float:
    """Computes degrees for deskewing using atan2."""
    return math.degrees(math.atan2(dy, dx))


def is_color_similar(color1: Tuple[int, int, int], color2: Tuple[int, int, int], threshold: int) -> bool:
    """Euclidean distance check for pixel similarity against a squared threshold."""
    return sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)) < threshold**2
