"""
Evaluates the button, if any, being pressed on a given frame
"""

import json
import logging
from itertools import product
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from config import CONFIG, logger
from debug_plotter import plot
from read_shadows import evaluate_shadow, find_element

img = cv2.imread(CONFIG.input_img_path, cv2.IMREAD_UNCHANGED)

if img is None:
    raise ValueError("Got None for image")

for template in CONFIG.templates:
    for idx, (image, px_roi) in enumerate(product(template.images, template.px_rois)):
        h_inp, w_inp = image.shape[:2]

        img_h, img_w = img.shape[:2]
        expanded_x1, expanded_y1, expanded_x2, expanded_y2 = px_roi.expanded_bounds(
            CONFIG.expand_roi_factor
        )

        y1, y2 = max(0, expanded_y1), min(img_h, expanded_y2)
        x1, x2 = max(0, expanded_x1), min(img_w, expanded_x2)

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

        found_target, rel_top_left, max_val, match = find_element(
            roi_img, image, CONFIG.matching_method
        )
        top_left = (rel_top_left[0] + x1, rel_top_left[1] + y1)

        if found_target:
            clicked = NotImplemented("Add color distance logic")
            logger.info(
                f"FOUND {template.name} at {top_left} with confidence of {max_val:.4f}; Clicked={clicked}"
            )
        else:
            logger.debug(
                f"Did not find {template.name} in ROI #{+1}; highest confidence of {max_val:.4f}"
            )
            clicked = False

        if CONFIG.debug_plot:
            plot(
                img=img,
                template=image,
                roi=(x1, y1, x2, y2),
                found_target=found_target,
                match=match,
                top_left=top_left,
                max_val=max_val,
                clicked=clicked,
                label=template.name,
            )

        if found_target:
            break
