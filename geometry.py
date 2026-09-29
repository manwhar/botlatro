from dataclasses import dataclass

import numpy as np


@dataclass
class NormalizedROI(frozen=True, slots=True):
    """Saves ROIs (Regions of Interest) as screen size ratios"""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init(self) -> None:
        bounds = (
            self.x_min,
            self.y_min,
            self.x_max,
            self.y_max,
        )
        if any(c <= 0 for c in bounds):
            raise ValueError(f"At least one invalid ROI boundary: {bounds}.")

        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("ROI minimums must be less than maximums.")


@dataclass
class PixelROI(frozen=True, slots=True):
    """Saves ROIs (Regions of Interest) as specific pixel counts"""

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def __post_init__(self) -> None:
        bounds = (
            self.x_min,
            self.y_min,
            self.x_max,
            self.y_max,
        )
        if any(c <= 0 for c in bounds):
            raise ValueError(f"At least one invalid ROI boundary: {bounds}.")

        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError("ROI minimums must be less than maximums.")

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def top_left(self) -> tuple[int, int]:
        return self.x_min, self.y_min

    @property
    def bottom_right(self) -> tuple[int, int]:
        return self.x_max, self.y_max


@dataclass(frozen=True, slots=True)
class ImageSize:
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Image dimensions must be positive")

    @classmethod
    def from_image(cls, image: np.ndarray) -> "ImageSize":
        height, width = image.shape[:2]
        return cls(width=width, height=height)
