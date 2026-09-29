import logging

import cv2

from geometry import ImageSize


class Settings:
    def __init__(self):
        self.template_json_path = "assets/templates.json"
        self.template_img_dir = "assets/templates"
        self.input_img_path = "assets/test_inputs/select_left_shop_item.jpg"
        self.base_resolution = ImageSize(width=1920, height=1080)
        self.matching_method = cv2.TM_CCORR_NORMED
        self.elem_conf_thresh = 0.95  # found element? confidence threshold

        self.colors = {  # BRG colors
            "sel_blind_gray": (64, 61, 44),
            "sel_blind_drop_shadow": (36, 34, 21),
        }

        self.debug_plot = False

        logging.basicConfig(
            level=logging.INFO,
            format="[%(levelname)s] %(message)s",
        )
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)


SETTINGS = Settings()
