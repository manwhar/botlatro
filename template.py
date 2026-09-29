import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cv2
from cv2.typing import MatLike
from geometry import ImageSize, NormalizedROI, PixelROI


@dataclass
class Template:
    name: str
    filenames: list[str]
    norm_rois: list[NormalizedROI]
    px_rois: list[PixelROI]
    images: list[MatLike]


def template_loader(
    template_json_path: str, template_img_dir: str, resolution: ImageSize
) -> list[Template]:
    """Load all templates from json."""
    with open(template_json_path, "r") as f:
        TEMPLATES: dict[str, dict[str, Any]] = json.load(f)

    templates = []

    for name, data in TEMPLATES.items():
        filenames, rois = data["filenames"], data["rois"]
        filenames: list[str]
        rois: list[list[float]]

        norm_rois = []
        px_rois = []
        for roi in rois:
            normalized_roi = NormalizedROI(*roi)
            norm_rois.append(normalized_roi)
            px_rois.append(
                PixelROI(
                    round(normalized_roi.x_min * resolution.width),
                    round(normalized_roi.y_min * resolution.height),
                    round(normalized_roi.x_max * resolution.width),
                    round(normalized_roi.y_max * resolution.height),
                )
            )

        images = [
            img
            for f in filenames
            if (img := cv2.imread(f"{template_img_dir}/{f}", cv2.IMREAD_UNCHANGED))
            is not None
        ]

        templates.append(Template(name, filenames, norm_rois, px_rois, images))

    return templates
