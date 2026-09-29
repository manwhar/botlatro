import logging

import cv2
from geometry import ImageSize
from template import template_loader


class Config:
    def __init__(self):
        self.template_json_path = "assets/templates.json"
        self.template_img_dir = "assets/templates"
        self.input_img_path = "assets/test_inputs/select_left_shop_item.jpg"
        self.resolution = ImageSize(width=1920, height=1080)
        self.matching_method = cv2.TM_CCORR_NORMED
        self.elem_conf_thresh = 0.95  # found element? confidence threshold
        self.expand_roi_factor = 1.3
        self.templates = template_loader(
            self.template_json_path, self.template_img_dir, self.resolution
        )
        self.debug_plot = True

        BRIGHT_GREEN = (133, 189, 52)
        GREEN_SHADOW = (21, 16, 12)
        self.colors = {  # BGR colors
            "sel_blind_backdrop": (64, 61, 44),
            "sel_blind_drop_shadow": (36, 34, 21),
            "cash_out_drop_shadow": (21, 19, 13),
            "cash_out_backdrop": (44, 43, 29),
            "reroll_backdrop": (40, 38, 27),
            "reroll_drop_shadow": (21, 16, 12),
            "next_round_drop_shadow": GREEN_SHADOW,
            "next_round_backdrop": BRIGHT_GREEN,  # NOTE: this is Reroll button green
        }

        logging.basicConfig(
            level=logging.INFO,
            format="[%(levelname)s] %(message)s",
        )
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)


CONFIG = Config()
logger = logging.getLogger(__name__)
