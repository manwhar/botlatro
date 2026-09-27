"""
Evaluates the button, if any, being pressed on a given frame
"""

from typing import Any
import logging

import cv2
import numpy as np
import json

import matplotlib.pyplot as plt

from read_shadows import COLORS, METHOD, find_element, evaluate_shadow

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

TEMPLATE_DATA_PATH = r"assets/templates.json"
IMAGE_PATH = r"assets/test_inputs/generic_blind_1080p.png"
TEMPLATE_FOLDER = r"assets/templates"
BASE_WIDTH, BASE_HEIGHT = 1920, 1080
PLOT_FLAG = False


def _to_rgb(image: np.ndarray) -> np.ndarray:
    """Helper to convert BGR/BGRA/Grayscale to RGB/RGBA for matplotlib."""
    if image is None:
        return image
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def plot(
    img: np.ndarray,
    template: np.ndarray,
    roi: tuple[int, int, int, int],
    found_target: bool,
    match: np.ndarray,
    top_left: tuple[int, int] | None = None,
    max_val: float | None = None,
    clicked: bool | None = None,
    label: str = "",
) -> None:
    """Plots visual debug data without relying on any global variables.

    Displays:
    - Full image with blue box for ROI and red box for detected element
    - Zoomed-in ROI with red box if found
    - cv2.matchTemplate response heatmap with colorbar
    - Template image
    - Cropped view of the detected element (if found)
    - Cropped view of shadow region with clicked status (if found)
    """
    x1, y1, x2, y2 = roi
    h_inp, w_inp = template.shape[:2]

    annotated_full = img.copy()
    # Blue box around ROI in BGR: (255, 0, 0)
    cv2.rectangle(annotated_full, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=3)

    bottom_right = None
    if found_target and top_left is not None:
        bottom_right = (top_left[0] + w_inp, top_left[1] + h_inp)
        # Red box around element in BGR: (0, 0, 255)
        cv2.rectangle(annotated_full, top_left, bottom_right, color=(0, 0, 255), thickness=3)

    # Zoomed ROI slice
    roi_crop = img[y1:y2, x1:x2].copy()
    if found_target and top_left is not None:
        rel_top_left = (top_left[0] - x1, top_left[1] - y1)
        rel_bottom_right = (rel_top_left[0] + w_inp, rel_top_left[1] + h_inp)
        cv2.rectangle(roi_crop, rel_top_left, rel_bottom_right, color=(0, 0, 255), thickness=2)

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))

    # Overall title
    title_parts = []
    if label:
        title_parts.append(f"Label: {label}")
    title_parts.append(f"Found: {found_target}")
    if max_val is not None:
        title_parts.append(f"Confidence: {max_val:.4f}")
    if clicked is not None:
        title_parts.append(f"Clicked: {clicked}")
    fig.suptitle(" | ".join(title_parts), fontsize=13, fontweight="bold")

    # [0, 0] Full Frame
    axs[0, 0].imshow(_to_rgb(annotated_full))
    axs[0, 0].set_title("Full Frame (Blue: ROI, Red: Target)")

    # [0, 1] Zoomed ROI
    axs[0, 1].imshow(_to_rgb(roi_crop))
    axs[0, 1].set_title(f"ROI Zoom [({x1}, {y1}) -> ({x2}, {y2})]")

    # [0, 2] matchTemplate Score Map
    im = axs[0, 2].imshow(match, cmap="viridis")
    axs[0, 2].set_title("matchTemplate Score Map")
    fig.colorbar(im, ax=axs[0, 2], fraction=0.046, pad=0.04)

    # [1, 0] Template Image
    axs[1, 0].imshow(_to_rgb(template))
    axs[1, 0].set_title(f"Template ({w_inp}x{h_inp})")

    # [1, 1] Detected Element Crop
    if found_target and top_left is not None and bottom_right is not None:
        pad = 6
        cy1 = max(0, top_left[1] - pad)
        cy2 = min(img.shape[0], bottom_right[1] + pad)
        cx1 = max(0, top_left[0] - pad)
        cx2 = min(img.shape[1], bottom_right[0] + pad)
        elem_crop = img[cy1:cy2, cx1:cx2]
        axs[1, 1].imshow(_to_rgb(elem_crop))
        axs[1, 1].set_title(f"Target Crop at {top_left}")
    else:
        axs[1, 1].text(0.5, 0.5, "Target Not Found", ha="center", va="center", fontsize=11, color="gray")
        axs[1, 1].set_title("Target Crop")

    # [1, 2] Shadow Region Crop
    if found_target and top_left is not None:
        sr = (h_inp - 1, h_inp + 2, 20, w_inp - 20)
        abs_y_start = top_left[1] + sr[0]
        abs_y_end = top_left[1] + sr[1]
        abs_x_start = top_left[0] + sr[2]
        abs_x_end = top_left[0] + sr[3]

        if abs_x_end > abs_x_start and abs_y_end > abs_y_start and abs_y_end <= img.shape[0] and abs_x_end <= img.shape[1]:
            shadow_img = img.copy()
            cv2.rectangle(
                shadow_img,
                (abs_x_start, abs_y_start),
                (abs_x_end, abs_y_end),
                color=(0, 0, 255),
                thickness=1,
            )
            pad_x, pad_y = 15, 15
            sy1 = max(0, abs_y_start - pad_y)
            sy2 = min(img.shape[0], abs_y_end + pad_y)
            sx1 = max(0, abs_x_start - pad_x)
            sx2 = min(img.shape[1], abs_x_end + pad_x)
            shadow_crop = shadow_img[sy1:sy2, sx1:sx2]
            axs[1, 2].imshow(_to_rgb(shadow_crop))
            axs[1, 2].set_title(f"Shadow Region (Clicked={clicked})")
        else:
            axs[1, 2].text(0.5, 0.5, "Shadow bounds invalid", ha="center", va="center", color="gray")
            axs[1, 2].set_title("Shadow Region")
    else:
        axs[1, 2].text(0.5, 0.5, "Target Not Found", ha="center", va="center", fontsize=11, color="gray")
        axs[1, 2].set_title("Shadow Region")

    plt.tight_layout()
    plt.show()


with open(TEMPLATE_DATA_PATH, 'r') as f:
    TEMPLATES: dict[str, dict[str, Any]] = json.load(f)

img = cv2.imread(IMAGE_PATH, cv2.IMREAD_UNCHANGED)

if img is None:
    raise ValueError("Got None for image")

for filename, data in TEMPLATES.items():
    template = cv2.imread(f"{TEMPLATE_FOLDER}/{filename}", cv2.IMREAD_UNCHANGED)

    if template is None:
        raise ValueError("Got None for template")

    h_inp, w_inp = template.shape[:2]

    rois: list[list[float]] = data["rois"]
    
    for idx, roi in enumerate(rois):
        min_x_ratio, min_y_ratio, max_x_ratio, max_y_ratio = roi
        min_x_px = round(min_x_ratio * BASE_WIDTH)
        max_x_px = round(max_x_ratio * BASE_WIDTH)
        min_y_px = round(min_y_ratio * BASE_HEIGHT)
        max_y_px = round(max_y_ratio * BASE_HEIGHT)

        img_h, img_w = img.shape[:2]
        y1, y2 = max(0, min_y_px), min(img_h, max_y_px)
        x1, x2 = max(0, min_x_px), min(img_w, max_x_px)

        # ensure roi is large enough for template
        if (y2 - y1) < h_inp:
            diff = h_inp - (y2 - y1)
            y1 = max(0, y1 - diff // 2)
            y2 = y1 + h_inp
            if y2 > img_h:
                y2 = img_h
                y1 = max(0, y2 - h_inp)
        if (x2 - x1) < w_inp:
            diff = w_inp - (x2 - x1)
            x1 = max(0, x1 - diff // 2)
            x2 = x1 + w_inp
            if x2 > img_w:
                x2 = img_w
                x1 = max(0, x2 - w_inp)

        if (y2 - y1) < h_inp or (x2 - x1) < w_inp:
            logger.warning("ROI is too small and cannot be expanded!")
            continue

        roi_img = img[y1:y2, x1:x2]

        found_target, rel_top_left, max_val, match = find_element(roi_img, template, METHOD)
        top_left = (rel_top_left[0] + x1, rel_top_left[1] + y1)

        if found_target:
            clicked = evaluate_shadow(img, top_left, max_val, w_inp, h_inp)
            logger.info(
                f"FOUND {data['label']} at {top_left} with confidence of {max_val:.4f}; Clicked={clicked}"
            )
        else:
            logger.debug(
                f"Did not find {data['label']} in ROI #{idx+1}; highest confidence of {max_val:.4f}"
            )
            clicked = False
        
        if PLOT_FLAG:
            plot(
                img=img,
                template=template,
                roi=(x1, y1, x2, y2),
                found_target=found_target,
                match=match,
                top_left=top_left,
                max_val=max_val,
                clicked=clicked,
                label=data.get("label", ""),
            )

        if found_target:
            break
        

