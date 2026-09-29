import cv2

from config import Config, configure_logging
from eval_template import eval
from template import template_loader

if __name__ == "__main__":
    configure_logging()
    config = Config()
    templates = template_loader(
        config.template_json_path,
        config.template_img_dir,
        config.resolution,
    )
    img = cv2.imread(config.input_img_path, cv2.IMREAD_UNCHANGED)
    for template in templates:
        result = eval(template, img, config)
