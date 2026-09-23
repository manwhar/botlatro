"""
Evaluates the button, if any, being pressed on a given frame
"""

import logging

import cv2
import numpy as np
from read_shadows import COLORS, METHOD, find_element, get_avg_color_from_region

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

IMAGE_PATH = r"test_inputs/balatro_3.png"
TEMPLATE_FOLDER = r"templates"
BASE_WIDTH, BASE_HEIGHT = 1920, 1080
TEMPLATES = {
    "buy_and_use.png": "Buy and Use",
    "buy.png": "Buy Card",
    "cash_out_full.png": "Cash Out",
    "discard.png": "Discard",
    "open.png": "Open Pack",
    "options.png": "Open Options Menu",
    "play_hand.png": "Play Hand",
    "redeem.png": "Redeem Voucher",
    "reroll_boss.png": "Reroll Boss",
    "reroll.png": "Reroll Shop",
    "run_info.png": "Open Run Info Menu",
    "select_blind.png": "Select Blind",
    "select.png": "Select Card",
    "sell_1.png": "Sell Card",
    "sell_2.png": "Sell Card",
    "sell_3.png": "Sell Card",
    "skip_blind.png": "Skip Blind",
    "sort_rank.png": "Sort by Rank",
    "sort_suit.png": "Sort by Suit",
    "use.png": "Use Card",
}
img = cv2.imread(IMAGE_PATH, cv2.IMREAD_UNCHANGED)

if img is None:
    raise ValueError("Got None for image")

for filename in TEMPLATES:
    template = cv2.imread(f"{TEMPLATE_FOLDER}/{filename}", cv2.IMREAD_UNCHANGED)

    if template is None:
        raise ValueError("Got None for template")

    h_inp, w_inp = template.shape[:2]
    original = img.copy()
    found_target, top_left, max_val, match = find_element(img, template, METHOD)
    if found_target:
        bottom_right = (top_left[0] + w_inp, top_left[1] + h_inp)

        # shadow_region: y_start, y_end, x_start, x_end
        sr = (h_inp - 1, h_inp + 2, 20, w_inp - 20)

        avg_bgr = get_avg_color_from_region(
            img=img,
            y_start=sr[0],
            y_end=sr[1],
            x_start=sr[2],
            x_end=sr[3],
            anchor=top_left,
        )

        abs_y_start = top_left[1] + sr[0]
        abs_y_end = top_left[1] + sr[1]
        abs_x_start = top_left[0] + sr[2]
        abs_x_end = top_left[0] + sr[3]

        shadow_img = img.copy()  # shadow_img shows shadow region
        cv2.rectangle(img, top_left, bottom_right, color=(0, 0, 255), thickness=10)
        cv2.rectangle(
            shadow_img,
            (abs_x_start, abs_y_start),
            (abs_x_end, abs_y_end),
            color=(0, 0, 255),
            thickness=1,
        )

        # is button pressed? based on color
        avg_bgr_np = np.array(avg_bgr)
        sel_blind_gray_np = np.array(COLORS["sel_blind_gray"])
        sel_blind_drop_shadow_np = np.array(COLORS["sel_blind_drop_shadow"])

        no_shadow_dist = np.average(abs(avg_bgr_np - sel_blind_gray_np))
        shadow_dist = np.average(abs(avg_bgr_np - sel_blind_drop_shadow_np))

        clicked = no_shadow_dist < shadow_dist

        logger.debug(
            f"Found target at {top_left} with confidence of {max_val:.4f}; Clicked={clicked} (t={filename})"
        )
    else:
        logger.debug(
            f"Did not find target; highest confidence of {max_val:.4f} (t={filename})"
        )
        shadow_img = img
        clicked = False
