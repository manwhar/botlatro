import time
import logging

import cv2

from config import Config, configure_logging
from eval_template import eval
from frame import iter_video_frames
from template import template_loader

if __name__ == "__main__":
    configure_logging()

    logger = logging.getLogger(__name__)

    config = Config()
    templates = template_loader(
        config.template_json_path,
        config.template_img_dir,
        config.resolution,
    )

    start = time.perf_counter()
    for idx, video_frame in enumerate(
        iter_video_frames(
            config.input_video_path,
            config.resolution,
            config.skip_frames,
            start_timestamp_seconds=40,
            end_timestamp_seconds=42,
        )
    ):
        print(f"\nProcessing frame {idx}...")
        for template in templates:
            result = eval(template, video_frame.image, config)
