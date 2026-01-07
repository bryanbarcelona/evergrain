import logging
from typing import Tuple
from dataclasses import dataclass

import numpy as np
from PIL import Image


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

    # median_color: Tuple[float, float, float] = (245.0, 245.0, 245.0)
    # color_variation: Tuple[float, float, float] = (1.5, 1.5, 1.5)
    median_color: Tuple[float, float, float] = (252.0, 254.0, 251.0)
    color_variation: Tuple[float, float, float] = (2.1714, 2.0376, 2.2785)

    @classmethod
    def from_image(cls, image: Image.Image, dpi: int, precision: int = 4) -> "ScanBackground":
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
                img_rgb = image.convert("RGB") if image.mode not in ("RGB", "L", "P") else image
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
        color_keys = ["r", "g", "b"]
        values = dict(zip(color_keys, pixel_color))
        medians = dict(zip(color_keys, self.median_color))
        stds = dict(zip(color_keys, self.color_variation))

        return all(abs(medians[k] - values[k]) <= stds[k] * spread for k in color_keys)
