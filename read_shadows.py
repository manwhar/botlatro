import cv2
import matplotlib.pyplot as plt
import numpy as np

F_ELEM_CONF_THRESH = 0.95  # found element confidence threshold
PRESSED_CONF_THRESH = 0.9  # is the button pressed? confidence threshold

COLORS = {
    "sel_blind_gray": (64, 61, 44, 255),
    "sel_blind_drop_shadow": (36, 34, 21, 255),
}

# METHOD = cv2.TM_CCOEFF_NORMED
METHOD = cv2.TM_CCORR_NORMED


def find_element(img, bgra_template, method) -> tuple[bool, tuple, float, np.ndarray]:
    img_bgr = img[:, :, :3]
    bgr_template = bgra_template[:, :, :3]
    alpha_mask = bgra_template[:, :, 3]
    result = cv2.matchTemplate(img_bgr, bgr_template, method, mask=alpha_mask)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    top_left = tuple(max_loc)

    found_target = max_val > F_ELEM_CONF_THRESH
    return found_target, top_left, max_val, result


def get_avg_color_from_region(
    img: np.ndarray,
    y_start: int,
    y_end: int,
    x_start: int,
    x_end: int,
    anchor: tuple = (0, 0),
) -> tuple:
    anchor_x, anchor_y = anchor

    abs_y_start = y_start + anchor_y
    abs_y_end = anchor_y + y_end
    abs_x_start = anchor_x + x_start
    abs_x_end = anchor_x + x_end

    # region of interest
    roi = img[abs_y_start:abs_y_end, abs_x_start:abs_x_end]

    average_color_per_row = np.average(roi, axis=0)
    average_color = np.average(average_color_per_row, axis=0)

    # round colors & return
    return tuple(np.round(average_color).astype(int))


def plot(img, template, found_target):
    fig, axs = plt.subplots(2, 3)
    fig.suptitle(f"Clicked={clicked}")

    if found_target:
        shadow_img = img.copy()  # shadow_img shows shadow region
        cv2.rectangle(img, top_left, bottom_right, color=(0, 0, 255), thickness=10)
        cv2.rectangle(
            shadow_img,
            (abs_x_start, abs_y_start),
            (abs_x_end, abs_y_end),
            color=(0, 0, 255),
            thickness=1,
        )
        shadow_img = cv2.cvtColor(shadow_img, cv2.COLOR_BGR2RGB)

        pad_x = 15
        pad_y = 15

        crop_y_start = max(0, abs_y_start - pad_y)  # type:ignore
        crop_y_end = min(shadow_img.shape[0], abs_y_end + pad_y)  # type:ignore
        crop_x_start = max(0, abs_x_start - pad_x)  # type:ignore
        crop_x_end = min(shadow_img.shape[1], abs_x_end + pad_x)  # type:ignore

        shadow_crop = shadow_img[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
        axs[1, 2].imshow(shadow_crop)
        axs[1, 2].set_title("shadow region")

    original = img.copy()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)

    # plt.subplot(121)
    axs[0, 0].imshow(original)
    axs[0, 0].set_title("input")

    axs[0, 1].imshow(match)
    axs[0, 1].set_title("matchTemplate")

    axs[1, 0].imshow(template)
    axs[1, 0].set_title("template")

    axs[1, 1].imshow(img)
    axs[1, 1].set_title("button location")

    plt.show()


if __name__ == "__main__":
    img = cv2.imread("test_inputs/balatro_3.png", cv2.IMREAD_UNCHANGED)
    template = cv2.imread("templates/skip_blind.png", cv2.IMREAD_UNCHANGED)

    if (template is None) or (img is None):
        raise ValueError(
            f"Got None for img or template: img={img}; template={template}"
        )

    h_inp, w_inp = template.shape[:2]
    original = img.copy()
    found_target, top_left, max_val, match = find_element(img, template, METHOD)
    if found_target:
        print(f"Found target at {top_left} with confidence of {max_val}.")

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

        # is button pressed? based on color
        avg_bgr_np = np.array(avg_bgr)
        sel_blind_gray_np = np.array(COLORS["sel_blind_gray"])
        sel_blind_drop_shadow_np = np.array(COLORS["sel_blind_drop_shadow"])

        no_shadow_dist = np.average(abs(avg_bgr_np - sel_blind_gray_np))
        shadow_dist = np.average(abs(avg_bgr_np - sel_blind_drop_shadow_np))

        clicked = no_shadow_dist < shadow_dist

        print(f"Avg color of region={avg_bgr}")
        print(f"hover_dist={no_shadow_dist}; shadow_dist={shadow_dist}")
        print(f"Clicked={clicked}")

    else:
        print(f"Did not find target; highest confidence of {max_val}")
        clicked = False

    plot(img, template, found_target)
