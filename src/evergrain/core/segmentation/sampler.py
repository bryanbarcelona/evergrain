from typing import Iterator, Tuple

from PIL import Image

from evergrain.core.models.segmentation import PixelData
from evergrain.core.exceptions.segmentation import EdgeReachedException


class Direction:
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


class ImageSampler:
    def __init__(self, image: Image.Image, dpi: int, precision: int = 50):
        self.image = image
        self.width, self.height = image.size
        self._pixel_data = image.load()
        self.dpi = dpi
        self.precision = max(1, min(precision, dpi))
        self.step = int(self.dpi / self.precision)

    def __iter__(self) -> Iterator[Tuple[int, int, int, int, int]]:
        for pixel in self.traverse(Direction.DOWN, self.step, self.step):
            for right_pixel in self.traverse(Direction.RIGHT, pixel.x, pixel.y, self.step):
                yield (right_pixel.x, right_pixel.y, right_pixel.r, right_pixel.g, right_pixel.b)

    def update_image(self, image: Image.Image) -> None:
        self.image = image
        self._pixel_data = image.load()

    def traverse(self, direction: str, x: int, y: int, distance: int = 0, max_steps: int = 0) -> Iterator[PixelData]:
        if distance == 0:
            distance = self.step

        count = 0
        current_pixel = self._get_pixel(x, y)
        yield current_pixel

        while True:
            try:
                current_pixel = self._move_in_direction(current_pixel.x, current_pixel.y, direction, distance)
                yield current_pixel
                if max_steps:
                    count += 1
                    if count >= max_steps:
                        break
            except EdgeReachedException:
                return

    def get_adjacent_pixels(self, x: int, y: int, distance: int = 0) -> Iterator[Tuple[int, int, int, int, int]]:
        for direction in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
            try:
                pixel = self._move_in_direction(x, y, direction, distance)
                yield (pixel.x, pixel.y, pixel.r, pixel.g, pixel.b)
            except EdgeReachedException:
                continue

    def _move_in_direction(self, x: int, y: int, direction: str, distance: int) -> PixelData:
        if distance == 0:
            distance = self.step

        if direction == Direction.UP:
            return self._move_up(x, y, distance)
        elif direction == Direction.DOWN:
            return self._move_down(x, y, distance)
        elif direction == Direction.LEFT:
            return self._move_left(x, y, distance)
        elif direction == Direction.RIGHT:
            return self._move_right(x, y, distance)
        else:
            raise ValueError(f"Unknown direction: {direction}")

    def _move_up(self, x: int, y: int, distance: int) -> PixelData:
        if y <= distance:
            raise EdgeReachedException
        return self._get_pixel(x, y - distance)

    def _move_down(self, x: int, y: int, distance: int) -> PixelData:
        if y >= self.height - distance - 1:
            raise EdgeReachedException
        return self._get_pixel(x, y + distance)

    def _move_left(self, x: int, y: int, distance: int) -> PixelData:
        if x <= distance:
            raise EdgeReachedException
        return self._get_pixel(x - distance, y)

    def _move_right(self, x: int, y: int, distance: int) -> PixelData:
        if x >= self.width - distance - 1:
            raise EdgeReachedException
        return self._get_pixel(x + distance, y)

    def _get_pixel(self, x: int, y: int) -> PixelData:
        r, g, b = self._pixel_data[x, y][:3]
        return PixelData(x, y, r, g, b)
