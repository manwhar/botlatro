import cv2
import matplotlib.pyplot as plt
import numpy as np

from geometry import PixelROI


def _to_rgb(image: np.ndarray) -> np.ndarray:
    """Helper to convert BGR/BGRA/Grayscale to RGB/RGBA for matplotlib."""
    if image is None:
        return image
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def plot(
    img: np.ndarray,
    template: np.ndarray,
    roi: tuple[int, int, int, int],
    found_target: bool,
    match: np.ndarray,
    top_left: tuple[int, int] | None = None,
    max_val: float | None = None,
    clicked: bool | None = None,
    label: str = "",
    shadow_region: PixelROI | None = None,
    shadow_threshold: float = 1000.0,
) -> None:
    """Plots visual debug data without relying on any global variables.

    Displays:
    - Full image with blue box for ROI and red box for detected element
    - Zoomed-in ROI with red box if found
    - cv2.matchTemplate response heatmap with colorbar
    - Template image
    - Cropped view of the detected element (if found)
    - Cropped view of shadow region with clicked status (if found)
    """
    x1, y1, x2, y2 = roi
    h_inp, w_inp = template.shape[:2]

    annotated_full = img.copy()
    # Blue box around ROI in BGR: (255, 0, 0)
    cv2.rectangle(annotated_full, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=3)

    bottom_right = None
    if found_target and top_left is not None:
        bottom_right = (top_left[0] + w_inp, top_left[1] + h_inp)
        # Red box around element in BGR: (0, 0, 255)
        cv2.rectangle(
            annotated_full, top_left, bottom_right, color=(0, 0, 255), thickness=3
        )

    # Zoomed ROI slice
    roi_crop = img[y1:y2, x1:x2].copy()
    if found_target and top_left is not None:
        rel_top_left = (top_left[0] - x1, top_left[1] - y1)
        rel_bottom_right = (rel_top_left[0] + w_inp, rel_top_left[1] + h_inp)
        cv2.rectangle(
            roi_crop, rel_top_left, rel_bottom_right, color=(0, 0, 255), thickness=2
        )

    fig, axs = plt.subplots(2, 4, figsize=(20, 8))
    axs[0, 3].axis("off")  # Leave top-right empty

    # Overall title
    title_parts = []
    if label:
        title_parts.append(f"Label: {label}")
    title_parts.append(f"Found: {found_target}")
    if max_val is not None:
        title_parts.append(f"Confidence: {max_val:.4f}")
    if clicked is not None:
        title_parts.append(f"Clicked: {clicked}")
    fig.suptitle(" | ".join(title_parts), fontsize=13, fontweight="bold")

    # [0, 0] Full Frame
    axs[0, 0].imshow(_to_rgb(annotated_full))
    axs[0, 0].set_title("Full Frame (Blue: ROI, Red: Target)")

    # [0, 1] Zoomed ROI
    axs[0, 1].imshow(_to_rgb(roi_crop))
    axs[0, 1].set_title(f"ROI Zoom [({x1}, {y1}) -> ({x2}, {y2})]")

    # [0, 2] matchTemplate Score Map
    im = axs[0, 2].imshow(match, cmap="viridis")
    axs[0, 2].set_title("matchTemplate Score Map")
    fig.colorbar(im, ax=axs[0, 2], fraction=0.046, pad=0.04)

    # [1, 0] Template Image
    axs[1, 0].imshow(_to_rgb(template))
    axs[1, 0].set_title(f"Template ({w_inp}x{h_inp})")

    # [1, 1] Detected Element Crop
    if found_target and top_left is not None and bottom_right is not None:
        pad = 6
        cy1 = max(0, top_left[1] - pad)
        cy2 = min(img.shape[0], bottom_right[1] + pad)
        cx1 = max(0, top_left[0] - pad)
        cx2 = min(img.shape[1], bottom_right[0] + pad)
        elem_crop = img[cy1:cy2, cx1:cx2]
        axs[1, 1].imshow(_to_rgb(elem_crop))
        axs[1, 1].set_title(f"Target Crop at {top_left}")
    else:
        axs[1, 1].text(
            0.5,
            0.5,
            "Target Not Found",
            ha="center",
            va="center",
            fontsize=11,
            color="gray",
        )
        axs[1, 1].set_title("Target Crop")

    # [1, 2] Shadow Region Crop & [1, 3] Edge Profile
    if found_target and top_left is not None:
        if shadow_region is not None:
            # Use provided shadow_region (PixelROI)
            abs_x_start = shadow_region.x_min
            abs_x_end = shadow_region.x_max
            abs_y_start = shadow_region.y_min
            abs_y_end = shadow_region.y_max
        else:
            # Fallback to old hardcoded logic if not provided
            sr = (h_inp - 1, h_inp + 2, 20, w_inp - 20)
            abs_y_start = top_left[1] + sr[0]
            abs_y_end = top_left[1] + sr[1]
            abs_x_start = top_left[0] + sr[2]
            abs_x_end = top_left[0] + sr[3]

        if (
            abs_x_end > abs_x_start
            and abs_y_end > abs_y_start
            and abs_y_end <= img.shape[0]
            and abs_x_end <= img.shape[1]
        ):
            shadow_img = img.copy()
            cv2.rectangle(
                shadow_img,
                (abs_x_start, abs_y_start),
                (abs_x_end, abs_y_end),
                color=(0, 0, 255),
                thickness=1,
            )
            pad_x, pad_y = 15, 15
            sy1 = max(0, abs_y_start - pad_y)
            sy2 = min(img.shape[0], abs_y_end + pad_y)
            sx1 = max(0, abs_x_start - pad_x)
            sx2 = min(img.shape[1], abs_x_end + pad_x)
            shadow_crop_padded = shadow_img[sy1:sy2, sx1:sx2]
            axs[1, 2].imshow(_to_rgb(shadow_crop_padded))
            axs[1, 2].set_title(f"Shadow Region (Clicked={clicked})")

            # [1, 3] Plot Edge Profile
            from shadow_evaluator import detect_horizontal_edges

            actual_shadow_crop = img[abs_y_start:abs_y_end, abs_x_start:abs_x_end]
            row_sums = detect_horizontal_edges(actual_shadow_crop)

            if len(row_sums) > 0:
                y_positions = np.arange(len(row_sums))
                axs[1, 3].plot(row_sums, y_positions, color="blue", linewidth=2)
                axs[1, 3].invert_yaxis()  # Match image coordinates (y goes down)
                axs[1, 3].set_title("Shadow Y-Edge Profile (Row Sums)")
                axs[1, 3].set_xlabel("Gradient Sum")
                axs[1, 3].set_ylabel("Row Index")
                axs[1, 3].axvline(
                    x=shadow_threshold,
                    color="red",
                    linestyle="--",
                    label="Default Threshold",
                )
                axs[1, 3].legend()
            else:
                axs[1, 3].text(
                    0.5, 0.5, "No edge data", ha="center", va="center", color="gray"
                )
                axs[1, 3].set_title("Shadow Edge Profile")
        else:
            for ax in (axs[1, 2], axs[1, 3]):
                ax.text(
                    0.5,
                    0.5,
                    "Shadow bounds invalid",
                    ha="center",
                    va="center",
                    color="gray",
                )
                ax.set_title("Shadow Analysis")
    else:
        for ax in (axs[1, 2], axs[1, 3]):
            ax.text(
                0.5,
                0.5,
                "Target Not Found",
                ha="center",
                va="center",
                fontsize=11,
                color="gray",
            )
            ax.set_title("Shadow Analysis")

    plt.tight_layout()

    # Maximize window to prevent drifting and ensure it's large enough
    try:
        manager = plt.get_current_fig_manager()
        backend = plt.get_backend()
        if backend == "TkAgg":
            manager.window.state("zoomed")
        elif backend == "wxAgg":
            manager.frame.Maximize(True)
        elif backend in ["Qt5Agg", "QtAgg", "Qt4Agg"]:
            manager.window.showMaximized()
    except Exception:  # noqa: BLE001, S110
        pass

    plt.show()
