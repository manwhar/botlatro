import logging

import cv2

from geometry import ImageSize


class Config:
    def __init__(self):
        self.template_json_path = "assets/templates/drspectred/templates.json"
        self.template_img_dir = "assets/templates/drspectred"
        self.input_img_path = "assets/test_inputs/scraped_frames/test_frame_1.jpg"
        self.input_video_path = "assets/test_inputs/videos/bu_full.mp4"
        self.resolution = ImageSize(width=1920, height=1080)
        self.matching_method = cv2.TM_CCORR_NORMED

        self.elem_conf_thresh = 0.95  # found element? confidence threshold
        self.expand_roi_factor = 1.3
        self.shadow_region_height = 10  # pixels
        self.shadow_region_vertical_shift = 2  # pixels
        self.shadow_threshold = 6000
        self.skip_frames = 2  # skip n frames then use 1

        self.debug_plot = False


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
