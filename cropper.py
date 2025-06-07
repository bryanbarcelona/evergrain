'''
Module: cropper.py
Author: [Your Name or Org]
Description:
    Provides high-precision background detection and photo segmentation
    from scanned images. The public API consists of:
        - ScanBackground: Background modeling for scan noise removal
        - PhotoSplitter: Extracts and deskews individual photos
'''

__all__ = ["ScanBackground", "PhotoSplitter"]

import logging
from collections import deque
from dataclasses import dataclass
from enum import Enum
from math import atan2, degrees
from typing import Iterator, Tuple, Set, List

import numpy as np
from PIL import Image
from PIL.Image import Resampling


# --- Logging Configuration ---

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# --- Enums and Exceptions ---

class _Direction(Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


class _EdgeReachedException(StopIteration):
    pass


# --- Data Classes ---

@dataclass(frozen=True)
class _PixelData:
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
class _ImageRegion:
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

    def calculate_overlap_ratio(self, other: '_ImageRegion') -> float:
        if (self.top > other.bottom or self.bottom < other.top or
            self.right < other.left or self.left > other.right):
            return 0.0

        overlap_width = min(self.right, other.right) - max(self.left, other.left)
        overlap_height = min(self.bottom, other.bottom) - max(self.top, other.top)
        overlap_area = overlap_width * overlap_height

        smaller_area = min(self.area, other.area)
        return float(overlap_area) / smaller_area if smaller_area > 0 else 0.0

    def try_merge_with(self, other: '_ImageRegion', threshold: float = 0.15) -> bool:
        if self.calculate_overlap_ratio(other) >= threshold:
            self.merge_with(other)
            return True
        return False

    def merge_with(self, other: '_ImageRegion') -> None:
        self.top = min(self.top, other.top)
        self.bottom = max(self.bottom, other.bottom)
        self.left = min(self.left, other.left)
        self.right = max(self.right, other.right)

    def is_larger_than(self, min_area: int) -> bool:
        return self.area > min_area

    def contains_point(self, x: int, y: int) -> bool:
        return self.left <= x <= self.right and self.top <= y <= self.bottom


# --- Image Sampling ---

class _ImageSampler:
    def __init__(self, image: Image.Image, dpi: int, precision: int = 50):
        self.image = image
        self.width, self.height = image.size
        self._pixel_data = image.load()
        self.dpi = dpi
        self.precision = max(1, min(precision, dpi))
        self.step = int(self.dpi / self.precision)

    def __iter__(self) -> Iterator[Tuple[int, int, int, int, int]]:
        for pixel in self.traverse(_Direction.DOWN, self.step, self.step):
            for right_pixel in self.traverse(_Direction.RIGHT, pixel.x, pixel.y, self.step):
                yield (right_pixel.x, right_pixel.y, right_pixel.r, right_pixel.g, right_pixel.b)

    def update_image(self, image: Image.Image) -> None:
        self.image = image
        self._pixel_data = image.load()

    def traverse(self, direction: _Direction, x: int, y: int, 
                 distance: int = 0, max_steps: int = 0) -> Iterator[_PixelData]:
        if distance == 0:
            distance = self.step

        count = 0
        current_pixel = self._get_pixel(x, y)
        yield current_pixel

        while True:
            try:
                current_pixel = self._move_in_direction(current_pixel.x, current_pixel.y, 
                                                       direction, distance)
                yield current_pixel
                if max_steps:
                    count += 1
                    if count >= max_steps:
                        break
            except _EdgeReachedException:
                return

    def get_adjacent_pixels(self, x: int, y: int, distance: int = 0) -> Iterator[Tuple[int, int, int, int, int]]:
        for direction in _Direction:
            try:
                pixel = self._move_in_direction(x, y, direction, distance)
                yield (pixel.x, pixel.y, pixel.r, pixel.g, pixel.b)
            except _EdgeReachedException:
                continue

    def _move_in_direction(self, x: int, y: int, direction: _Direction, distance: int) -> _PixelData:
        if distance == 0:
            distance = self.step

        handlers = {
            _Direction.UP: self._move_up,
            _Direction.DOWN: self._move_down,
            _Direction.LEFT: self._move_left,
            _Direction.RIGHT: self._move_right,
        }
        return handlers[direction](x, y, distance)

    def _move_up(self, x: int, y: int, distance: int) -> _PixelData:
        if y <= distance:
            raise _EdgeReachedException
        return self._get_pixel(x, y - distance)

    def _move_down(self, x: int, y: int, distance: int) -> _PixelData:
        if y >= self.height - distance - 1:
            raise _EdgeReachedException
        return self._get_pixel(x, y + distance)

    def _move_left(self, x: int, y: int, distance: int) -> _PixelData:
        if x <= distance:
            raise _EdgeReachedException
        return self._get_pixel(x - distance, y)

    def _move_right(self, x: int, y: int, distance: int) -> _PixelData:
        if x >= self.width - distance - 1:
            raise _EdgeReachedException
        return self._get_pixel(x + distance, y)

    def _get_pixel(self, x: int, y: int) -> _PixelData:
        r, g, b = self._pixel_data[x, y][:3]
        return _PixelData(x, y, r, g, b)

# --- Edge Detection ---

class _EdgeDetector:
    PRECISION = 6
    SAMPLE_COUNT = PRECISION - 2

    def __init__(self, sampler: _ImageSampler):
        self.sampler = sampler
        self.width = sampler.width
        self.height = sampler.height


class _TopEdgeDetector(_EdgeDetector):
    def __init__(self, sampler: _ImageSampler):
        super().__init__(sampler)
        self.step_size = self.width / self.PRECISION
        self.start_x = self.step_size
        self.start_y = 0

    def get_parallel_samples(self) -> Iterator[_PixelData]:
        return self.sampler.traverse(
            _Direction.RIGHT, int(self.start_x), int(self.start_y), 
            int(self.step_size), self.SAMPLE_COUNT
        )

    def get_perpendicular_samples(self, x: int, y: int) -> Iterator[_PixelData]:
        return self.sampler.traverse(_Direction.DOWN, x, y, 1)

    def get_distance(self, x: int, y: int) -> int:
        return y

    def calculate_angle(self, prev_distance: int, x: int, y: int) -> float:
        return atan2(y - prev_distance, self.step_size)


class _RightEdgeDetector(_TopEdgeDetector):
    def __init__(self, sampler: _ImageSampler):
        super().__init__(sampler)
        self.step_size = self.height / self.PRECISION
        self.start_x = sampler.width - 1
        self.start_y = self.step_size

    def get_parallel_samples(self) -> Iterator[_PixelData]:
        return self.sampler.traverse(
            _Direction.DOWN, int(self.start_x), int(self.start_y),
            int(self.step_size), self.SAMPLE_COUNT
        )

    def get_perpendicular_samples(self, x: int, y: int) -> Iterator[_PixelData]:
        return self.sampler.traverse(_Direction.LEFT, x, y, 1)

    def get_distance(self, x: int, y: int) -> int:
        return x

    def calculate_angle(self, prev_distance: int, x: int, y: int) -> float:
        return atan2(prev_distance - x, self.step_size)


class _BottomEdgeDetector(_TopEdgeDetector):
    def __init__(self, sampler: _ImageSampler):
        super().__init__(sampler)
        self.step_size = self.width / self.PRECISION
        self.start_x = sampler.width - self.step_size
        self.start_y = sampler.height - 1

    def get_parallel_samples(self) -> Iterator[_PixelData]:
        return self.sampler.traverse(
            _Direction.LEFT, int(self.start_x), int(self.start_y),
            int(self.step_size), self.SAMPLE_COUNT
        )

    def get_perpendicular_samples(self, x: int, y: int) -> Iterator[_PixelData]:
        return self.sampler.traverse(_Direction.UP, x, y, 1)

    def get_distance(self, x: int, y: int) -> int:
        return y

    def calculate_angle(self, prev_distance: int, x: int, y: int) -> float:
        return atan2(prev_distance - y, self.step_size)


class _LeftEdgeDetector(_TopEdgeDetector):
    def __init__(self, sampler: _ImageSampler):
        super().__init__(sampler)
        self.step_size = self.height / self.PRECISION
        self.start_x = 0
        self.start_y = sampler.height - self.step_size

    def get_parallel_samples(self) -> Iterator[_PixelData]:
        return self.sampler.traverse(
            _Direction.UP, int(self.start_x), int(self.start_y),
            int(self.step_size), self.SAMPLE_COUNT
        )

    def get_perpendicular_samples(self, x: int, y: int) -> Iterator[_PixelData]:
        return self.sampler.traverse(_Direction.RIGHT, x, y, 1)

    def get_distance(self, x: int, y: int) -> int:
        return x

    def calculate_angle(self, prev_distance: int, x: int, y: int) -> float:
        return atan2(x - prev_distance, self.step_size)

# --- Deskewing ---

@dataclass
class _DeskewResult:
    image: Image.Image
    margins: Tuple[int, int, int, int]
    rotation_angle: float


class _PhotoDeskewer:
    def __init__(self, image: Image.Image, background: 'ScanBackground', contrast: int = 10, shrink: int = 0):
        self.image = image
        self.width, self.height = image.size
        self.background = background
        self.contrast = contrast
        self.shrink = shrink

        sampler = _ImageSampler(image, dpi=1, precision=1)
        self.edge_detectors = [
            _LeftEdgeDetector(sampler),
            _TopEdgeDetector(sampler),
            _RightEdgeDetector(sampler),
            _BottomEdgeDetector(sampler),
        ]

    def correct_skew(self) -> _DeskewResult:
        margin_angle_pairs = [self._analyze_edge(d) for d in self.edge_detectors]
        margins, angles = zip(*margin_angle_pairs)

        rotation_angle = degrees(np.median(angles))
        rotated_image = self.image.rotate(rotation_angle, Resampling.BICUBIC)

        adjusted_margins = (
            margins[0] + self.shrink,
            margins[1] + self.shrink,
            margins[2] - self.shrink,
            margins[3] - self.shrink,
        )

        cropped = rotated_image.crop(adjusted_margins)
        return _DeskewResult(cropped, adjusted_margins, rotation_angle)

    def _analyze_edge(self, detector: _EdgeDetector) -> Tuple[int, float]:
        distances = []
        angles = []

        for start in detector.get_parallel_samples():
            samples = detector.get_perpendicular_samples(start.x, start.y)
            x, y = start.x, start.y

            for p in samples:
                if self.background.matches(p.color, self.contrast):
                    break
                if detector.get_distance(p.x, p.y) > detector.step_size:
                    samples = detector.get_perpendicular_samples(start.x, start.y)
                    break

            for p in samples:
                if not self.background.matches(p.color, self.contrast):
                    break
                x, y = p.x, p.y

            if distances:
                angles.append(detector.calculate_angle(distances[-1], x, y))

            distances.append(detector.get_distance(x, y))

        return int(np.median(distances)), np.median(angles)

# --- Public API: ScanBackground ---

@dataclass
class ScanBackground:
    """
    Represents the background color profile of a scanned image.

    This class is used to characterize the typical background color and color 
    variation of a scanner flatbed, enabling distinction between the scanned 
    photos and the background.

    Attributes:
        median_color (Tuple[float, float, float]): Median RGB color values of the background.
        color_variation (Tuple[float, float, float]): Standard deviation of RGB values indicating color variation.
    """

    median_color: Tuple[float, float, float] = (245.0, 245.0, 245.0)
    color_variation: Tuple[float, float, float] = (1.5, 1.5, 1.5)

    @classmethod
    def from_image(cls, image: Image.Image, dpi: int, precision: int = 4) -> 'ScanBackground':
        """
        Creates a ScanBackground profile by sampling pixels from the given image.

        Samples pixels uniformly across the image based on the dpi and precision parameters,
        computes the median and standard deviation of the RGB values, and returns a new
        ScanBackground instance representing the typical background color profile.

        Args:
            image (Image.Image): The scanned image to sample.
            dpi (int): The scanning resolution in dots per inch.
            precision (int, optional): Number of sampling steps per dpi; controls sampling density. Defaults to 4.

        Returns:
            ScanBackground: A background color profile with median color and color variation.
        """
        if not image:
            logging.error("ScanBackground.from_image received None image.")
            return cls()

        precision = min(max(precision, 1), dpi)
        step = int(dpi / precision)
        pixels_rgb = []

        for y in range(step, image.height, step):
            for x in range(step, image.width, step):
                img_rgb = image.convert('RGB') if image.mode not in ('RGB', 'L', 'P') else image
                pixels_rgb.append(img_rgb.getpixel((x, y))[:3])

        if not pixels_rgb:
            logging.warning("No pixels sampled in ScanBackground.from_image.")
            return cls()

        array = np.array(pixels_rgb)
        median_c = tuple(np.median(array, axis=0))
        color_var = tuple(np.std(array, axis=0))
        return cls(median_color=median_c, color_variation=color_var)

    def matches(self, pixel_color: Tuple[int, int, int], spread: float) -> bool:
        """
        Determines if a pixel color matches the background color profile within a given tolerance.

        Args:
            pixel_color (Tuple[int, int, int]): RGB color of the pixel to compare.
            spread (float): Multiplier for color variation tolerance.

        Returns:
            bool: True if the pixel color is within the tolerated range of the background color; False otherwise.
        """
        color_keys = ['r', 'g', 'b']
        values = dict(zip(color_keys, pixel_color))
        medians = dict(zip(color_keys, self.median_color))
        stds = dict(zip(color_keys, self.color_variation))

        return all(abs(medians[k] - values[k]) <= stds[k] * spread for k in color_keys)


# --- Public API: PhotoSplitter ---

class PhotoSplitter:
    """
    Splits a scanned image containing multiple photos into individual deskewed images.

    Uses the background color profile to detect photo regions via flood fill, isolates
    those regions, and optionally deskews the resulting photo crops.

    Attributes:
        image (Image.Image): The scanned source image containing multiple photos.
        width (int): Width of the source image in pixels.
        height (int): Height of the source image in pixels.
        background (ScanBackground): Background color profile for distinguishing photos.
        dpi (int): Resolution of the scanned image in dots per inch.
        deskew_enabled (bool): Flag to enable deskewing of extracted photo regions.
        contrast (int): Threshold multiplier used for color matching tolerance.
        shrink (int): Factor to reduce resolution when deskewing.
        precision (int): Sampling precision used for scanning.
        sampler (_ImageSampler): Internal pixel sampler helper.
        photo_regions (List[_ImageRegion]): List of detected photo regions as bounding boxes.
    """

    def __init__(self, image: Image.Image, background_profile: ScanBackground, dpi: int,
                 sample_precision: int = 50, deskew: bool = True, contrast: int = 15, shrink: int = 3):
        """
        Initializes the PhotoSplitter with a scanned image and background profile.

        Args:
            image (Image.Image): The scanned image containing multiple photos.
            background_profile (ScanBackground): Background color profile for detection.
            dpi (int): Resolution of the scanned image in dots per inch.
            sample_precision (int, optional): Number of samples to take for detection. Defaults to 50.
            deskew (bool, optional): Whether to deskew detected photo regions. Defaults to True.
            contrast (int, optional): Contrast threshold factor for background matching. Defaults to 15.
            shrink (int, optional): Shrink factor used during deskewing. Defaults to 3.

        Raises:
            TypeError: If image is not a PIL Image instance.
            TypeError: If background_profile is not a ScanBackground instance.
            ValueError: If dpi is not a positive integer.
        """
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image")
        if not isinstance(background_profile, ScanBackground):
            raise TypeError("background_profile must be a ScanBackground instance")
        if not isinstance(dpi, int) or dpi <= 0:
            raise ValueError("dpi must be a positive integer")

        self.image = image
        self.width, self.height = image.size
        self.background = background_profile
        self.dpi = dpi
        self.deskew_enabled = deskew
        self.contrast = contrast
        self.shrink = shrink
        self.precision = sample_precision

        self.sampler = _ImageSampler(image, dpi, sample_precision)
        self.photo_regions = self._find_photo_regions()

    def __iter__(self) -> Iterator[Image.Image]:
        """
        Iterates over the detected photo regions, yielding each as a cropped and optionally deskewed image.

        Yields:
            Iterator[Image.Image]: Cropped photo images extracted from the scanned image.
        """
        for region in self.photo_regions:
            subimage = self.image.crop(region.bounds)
            if self.deskew_enabled:
                deskewer = _PhotoDeskewer(subimage, self.background, self.contrast, self.shrink)
                yield deskewer.correct_skew().image
            else:
                yield subimage

    def _find_photo_regions(self) -> List[_ImageRegion]:
        """
        Detects individual photo regions in the scanned image by sampling pixels and flood filling.

        Returns:
            List[_ImageRegion]: A list of bounding boxes for each detected photo region.
        """
        regions = []

        for x, y, r, g, b in self.sampler:
            if self.background.matches((r, g, b), self.contrast):
                continue
            if any(region.contains_point(x, y) for region in regions):
                continue

            connected = self._flood_fill((x, y, r, g, b))
            if connected:
                xs, ys = zip(*connected)
                new_region = _ImageRegion(min(xs), min(ys), max(xs), max(ys))
                merged = any(r.try_merge_with(new_region) for r in regions)
                if not merged:
                    regions.append(new_region)

        min_area = self.dpi ** 2
        return [r for r in regions if r.is_larger_than(min_area)]

    def _flood_fill(self, start_pixel: Tuple[int, int, int, int, int]) -> Set[Tuple[int, int]]:
        """
        Performs a flood fill from a starting pixel to find all connected pixels not matching the background.

        Args:
            start_pixel (Tuple[int, int, int, int, int]): Starting pixel coordinates and RGB color (x, y, r, g, b).

        Returns:
            Set[Tuple[int, int]]: Set of (x, y) coordinates belonging to the connected photo region.
        """
        x, y, r, g, b = start_pixel
        start_pos = (x, y)
        start_color = (r, g, b)
        to_visit = [start_pos]
        visited = set(to_visit)

        for pos in to_visit:
            px, py = pos
            for ax, ay, ar, ag, ab in self.sampler.get_adjacent_pixels(px, py):
                adj_pos = (ax, ay)
                if adj_pos not in visited:
                    visited.add(adj_pos)
                    if not self.background.matches((ar, ag, ab), self.contrast):
                        to_visit.append(adj_pos)

        return visited
