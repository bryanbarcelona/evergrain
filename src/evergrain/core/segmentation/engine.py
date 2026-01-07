from typing import Iterator, Tuple, Set, List

from PIL import Image

from evergrain.core.models.segmentation import ImageRegion
from evergrain.core.segmentation.sampler import ImageSampler
from evergrain.core.segmentation.deskew import PhotoDeskewer
from evergrain.core.segmentation.background import ScanBackground


class PhotoSplitter:
    def __init__(
        self,
        image: Image.Image,
        dpi: int,
        background_profile: ScanBackground = None,
        sample_precision: int = 50,
        deskew: bool = True,
        contrast: int = 15,
        shrink: int = 3,
    ):
        """
        Initializes the PhotoSplitter to detect and extract photos from a scan.

        Args:
            image (Image.Image): The scanned image to process.
            dpi (int): The scanning resolution in dots per inch.
            background_profile (ScanBackground, optional): The color profile of
                the scanner background. Defaults to calibrated factory values.
            sample_precision (int): Number of sampling steps for the image.
            deskew (bool): Whether to automatically straighten detected photos.
            contrast (int): Sensitivity multiplier for background detection.
            shrink (int): Pixel padding to remove from edges of detected photos.

        Raises:
            TypeError: If image or background_profile are the wrong types.
            ValueError: If dpi is not a positive integer.
        """

        self.background = background_profile or ScanBackground()

        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image")
        if not isinstance(self.background, ScanBackground):
            raise TypeError("background_profile must be a ScanBackground instance")
        if not isinstance(dpi, int) or dpi <= 0:
            raise ValueError("dpi must be a positive integer")

        self.image = image
        self.width, self.height = image.size
        self.dpi = dpi
        self.deskew_enabled = deskew
        self.contrast = contrast
        self.shrink = shrink
        self.precision = sample_precision

        self.sampler = ImageSampler(image, dpi, sample_precision)
        self.photo_regions = self._find_photo_regions()

    def __iter__(self) -> Iterator[Image.Image]:
        for region in self.photo_regions:
            subimage = self.image.crop(region.bounds)
            if self.deskew_enabled:
                deskewer = PhotoDeskewer(subimage, self.background, self.contrast, self.shrink)
                yield deskewer.correct_skew().image
            else:
                yield subimage

    def _find_photo_regions(self) -> List[ImageRegion]:
        regions = []

        for x, y, r, g, b in self.sampler:
            if self.background.matches((r, g, b), self.contrast):
                continue
            if any(region.contains_point(x, y) for region in regions):
                continue

            connected = self._flood_fill((x, y, r, g, b))
            if connected:
                xs, ys = zip(*connected)
                new_region = ImageRegion(min(xs), min(ys), max(xs), max(ys))
                merged = any(r.try_merge_with(new_region) for r in regions)
                if not merged:
                    regions.append(new_region)

        min_area = self.dpi**2
        return [r for r in regions if r.is_larger_than(min_area)]

    def _flood_fill(self, start_pixel: Tuple[int, int, int, int, int]) -> Set[Tuple[int, int]]:
        x, y, r, g, b = start_pixel
        start_pos = (x, y)
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
