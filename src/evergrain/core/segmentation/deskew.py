from math import degrees
from typing import Tuple

import numpy as np
from PIL import Image
from PIL.Image import Resampling

from evergrain.core.segmentation.sampler import ImageSampler
from evergrain.core.segmentation.edge_detectors import (
    LeftEdgeDetector,
    TopEdgeDetector,
    RightEdgeDetector,
    BottomEdgeDetector,
)
from evergrain.core.models.segmentation import DeskewResult
from evergrain.core.segmentation.background import ScanBackground


class PhotoDeskewer:
    def __init__(self, image: Image.Image, background: ScanBackground, contrast: int = 10, shrink: int = 0):
        self.image = image
        self.width, self.height = image.size
        self.background = background
        self.contrast = contrast
        self.shrink = shrink

        sampler = ImageSampler(image, dpi=1, precision=1)
        self.edge_detectors = [
            LeftEdgeDetector(sampler),
            TopEdgeDetector(sampler),
            RightEdgeDetector(sampler),
            BottomEdgeDetector(sampler),
        ]

    def correct_skew(self) -> DeskewResult:
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
        return DeskewResult(cropped, adjusted_margins, rotation_angle)

    def _analyze_edge(self, detector) -> Tuple[int, float]:
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
