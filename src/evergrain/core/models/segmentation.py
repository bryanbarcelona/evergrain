from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image


@dataclass(frozen=True)
class PixelData:
    x: int
    y: int
    r: int
    g: int
    b: int

    @property
    def position(self) -> Tuple[int, int]:
        return (self.x, self.y)

    @property
    def color(self) -> Tuple[int, int, int]:
        return (self.r, self.g, self.b)


@dataclass
class ImageRegion:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def bounds(self) -> Tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)

    def calculate_overlap_ratio(self, other: 'ImageRegion') -> float:
        if (self.top > other.bottom or self.bottom < other.top or
            self.right < other.left or self.left > other.right):
            return 0.0

        overlap_width = min(self.right, other.right) - max(self.left, other.left)
        overlap_height = min(self.bottom, other.bottom) - max(self.top, other.top)
        overlap_area = overlap_width * overlap_height

        smaller_area = min(self.area, other.area)
        return float(overlap_area) / smaller_area if smaller_area > 0 else 0.0

    def try_merge_with(self, other: 'ImageRegion', threshold: float = 0.15) -> bool:
        if self.calculate_overlap_ratio(other) >= threshold:
            self.merge_with(other)
            return True
        return False

    def merge_with(self, other: 'ImageRegion') -> None:
        self.top = min(self.top, other.top)
        self.bottom = max(self.bottom, other.bottom)
        self.left = min(self.left, other.left)
        self.right = max(self.right, other.right)

    def is_larger_than(self, min_area: int) -> bool:
        return self.area > min_area

    def contains_point(self, x: int, y: int) -> bool:
        return self.left <= x <= self.right and self.top <= y <= self.bottom


@dataclass
class DeskewResult:
    image: Image.Image
    margins: Tuple[int, int, int, int]
    rotation_angle: float