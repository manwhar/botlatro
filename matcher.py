from dataclasses import dataclass

import cv2
from cv2.typing import MatLike

from geometry import ImageSize, PixelROI


@dataclass(frozen=True, slots=True)
class TemplateMatch:
    found: bool
    top_left: tuple[int, int]
    confidence: float
    score_map: MatLike | None
    roi: PixelROI
    template_size: ImageSize


def find_element(
    img,
    bgra_template,
    method,
    roi: PixelROI,
    confidence_threshold: float,
) -> TemplateMatch:
    img_bgr = img[:, :, :3]
    bgr_template = bgra_template[:, :, :3]
    alpha_mask = bgra_template[:, :, 3]
    result: MatLike = cv2.matchTemplate(img_bgr, bgr_template, method, mask=alpha_mask)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    top_left = tuple(max_loc)

    found_target = max_val > confidence_threshold

    template_size = ImageSize.from_image(bgra_template)

    return TemplateMatch(found_target, top_left, max_val, result, roi, template_size)
