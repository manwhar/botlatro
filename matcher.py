from dataclasses import dataclass

import numpy as np

from geometry import ImageSize, PixelROI


@dataclass(frozen=True, slots=True)
class TemplateMatch:
    found: bool
    top_left: tuple[int, int]
    confidence: float
    score_map: np.ndarray | None
    roi: PixelROI
    template_size: ImageSize
