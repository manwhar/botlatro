import logging

import cv2

from geometry import ImageSize


class Config:
    def __init__(self):
        self.template_json_path = "assets/templates/drspectred/templates.json"
        self.template_img_dir = "assets/templates/drspectred"
        self.input_img_path = "assets/sample_inputs/scraped_frames/test_frame_1.jpg"
        self.input_video_path = "assets/sample_inputs/videos/bu_full.mp4"
        self.resolution = ImageSize(width=1920, height=1080)
        self.matching_method = cv2.TM_CCORR_NORMED

        self.elem_conf_thresh = 0.946  # found element? confidence threshold
        self.expand_roi_factor = 1.3
        self.shadow_region_height = 10  # pixels
        self.shadow_threshold = 52.0
        self.skip_frames = 20  # skip n frames then use 1

        self.start_timestamp = 80
        self.end_timestamp = 800

        self.debug_plot = False
        self.setup_mode = True


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
