import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def detect_horizontal_edges(image: np.ndarray) -> np.ndarray:
    """
    Runs a Y-directional Sobel edge detection to find horizontal lines,
    and returns the sum of gradients across each row.

    Args:
        image: A pre-sliced numpy array image (ROI).

    Returns:
        A 1D numpy array containing the sum of absolute Y-gradients for each row.
    """
    if image is None or image.size == 0:
        logger.warning("detect_horizontal_edges received an empty or None image.")
        return np.array([])

    height, width = image.shape[:2]

    # Shadow slices are expected to be thin and wide (aspect ratio W/H >> 1)
    if height == 0 or width == 0:
        logger.warning(f"Invalid image dimensions: {width}x{height}.")
    elif width / height <= 2:  # 2:1 aspect ratio or worse
        logger.warning(
            f"Unexpected aspect ratio for horizontal edge detection. Image is {width}x{height} (W <= H). Expected a wide, thin slice."
        )
    elif height < 3 or width < 10:
        logger.warning(
            f"Image slice might be too small for reliable edge detection: {width}x{height}."
        )

    # Convert to grayscale if not already
    if len(image.shape) == 3:
        if image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Run Sobel in Y direction (dx=0, dy=1) to isolate horizontal edges
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Take absolute value to get magnitude of gradients
    abs_sobel_y = np.abs(sobel_y)

    # Sum the gradients horizontally across each row
    row_sums = np.sum(abs_sobel_y, axis=1)

    return row_sums


def analyze_shadow_presence(image: np.ndarray, threshold: float = 1000.0) -> bool:
    """
    Analyzes a pre-sliced image to determine if a shadow is present,
    based on finding a strong horizontal edge.

    Args:
        image: A pre-sliced numpy array image directly below the button.
        threshold: The gradient sum threshold to consider a row as a valid shadow edge.
                   This may need to be tuned based on image width and contrast.

    Returns:
        True if a distinct horizontal shadow edge is found (unpressed), False otherwise.
    """
    row_sums = detect_horizontal_edges(image)

    if len(row_sums) == 0:
        return False

    max_gradient = np.max(row_sums)
    return bool(max_gradient > threshold)
