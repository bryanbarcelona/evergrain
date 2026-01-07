from math import atan2
from typing import Iterator

from evergrain.core.segmentation.sampler import ImageSampler, Direction
from evergrain.core.models.segmentation import PixelData


class EdgeDetector:
    PRECISION = 6
    SAMPLE_COUNT = PRECISION - 2

    def __init__(self, sampler: ImageSampler):
        self.sampler = sampler
        self.width = sampler.width
        self.height = sampler.height


class TopEdgeDetector(EdgeDetector):
    def __init__(self, sampler: ImageSampler):
        super().__init__(sampler)
        self.step_size = self.width / self.PRECISION
        self.start_x = self.step_size
        self.start_y = 0

    def get_parallel_samples(self) -> Iterator[PixelData]:
        return self.sampler.traverse(
            Direction.RIGHT, int(self.start_x), int(self.start_y), int(self.step_size), self.SAMPLE_COUNT
        )

    def get_perpendicular_samples(self, x: int, y: int) -> Iterator[PixelData]:
        return self.sampler.traverse(Direction.DOWN, x, y, 1)

    def get_distance(self, x: int, y: int) -> int:
        return y

    def calculate_angle(self, prev_distance: int, x: int, y: int) -> float:
        return atan2(y - prev_distance, self.step_size)


class RightEdgeDetector(TopEdgeDetector):
    def __init__(self, sampler: ImageSampler):
        super().__init__(sampler)
        self.step_size = self.height / self.PRECISION
        self.start_x = self.sampler.width - 1
        self.start_y = self.step_size

    def get_parallel_samples(self) -> Iterator[PixelData]:
        return self.sampler.traverse(
            Direction.DOWN, int(self.start_x), int(self.start_y), int(self.step_size), self.SAMPLE_COUNT
        )

    def get_perpendicular_samples(self, x: int, y: int) -> Iterator[PixelData]:
        return self.sampler.traverse(Direction.LEFT, x, y, 1)

    def get_distance(self, x: int, y: int) -> int:
        return x

    def calculate_angle(self, prev_distance: int, x: int, y: int) -> float:
        return atan2(prev_distance - x, self.step_size)


class BottomEdgeDetector(TopEdgeDetector):
    def __init__(self, sampler: ImageSampler):
        super().__init__(sampler)
        self.step_size = self.width / self.PRECISION
        self.start_x = self.sampler.width - self.step_size
        self.start_y = self.sampler.height - 1

    def get_parallel_samples(self) -> Iterator[PixelData]:
        return self.sampler.traverse(
            Direction.LEFT, int(self.start_x), int(self.start_y), int(self.step_size), self.SAMPLE_COUNT
        )

    def get_perpendicular_samples(self, x: int, y: int) -> Iterator[PixelData]:
        return self.sampler.traverse(Direction.UP, x, y, 1)

    def get_distance(self, x: int, y: int) -> int:
        return y

    def calculate_angle(self, prev_distance: int, x: int, y: int) -> float:
        return atan2(prev_distance - y, self.step_size)


class LeftEdgeDetector(TopEdgeDetector):
    def __init__(self, sampler: ImageSampler):
        super().__init__(sampler)
        self.step_size = self.height / self.PRECISION
        self.start_x = 0
        self.start_y = self.sampler.height - self.step_size

    def get_parallel_samples(self) -> Iterator[PixelData]:
        return self.sampler.traverse(
            Direction.UP, int(self.start_x), int(self.start_y), int(self.step_size), self.SAMPLE_COUNT
        )

    def get_perpendicular_samples(self, x: int, y: int) -> Iterator[PixelData]:
        return self.sampler.traverse(Direction.RIGHT, x, y, 1)

    def get_distance(self, x: int, y: int) -> int:
        return x

    def calculate_angle(self, prev_distance: int, x: int, y: int) -> float:
        return atan2(x - prev_distance, self.step_size)
