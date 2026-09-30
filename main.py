import logging
import time

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
            start_timestamp_seconds=config.start_timestamp,
            end_timestamp_seconds=config.end_timestamp,
        )
    ):
        print(f"\nProcessing frame {idx}...")
        sec = int(video_frame.timestamp_seconds)
        hh = sec // 3600
        mm = (sec % 3600) // 60
        ss = sec % 60
        ff = video_frame.index % 60
        ts_str = f"{video_frame.timestamp_seconds:.2f}s | {hh:02}:{mm:02}:{ss:02}:{ff:02}"
        
        for template in templates:
            result = eval(template, video_frame.image, config, video_timestamp_str=ts_str)
