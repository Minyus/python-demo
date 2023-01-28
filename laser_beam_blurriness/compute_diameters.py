import numpy as np
from generate_gaussian import generate_gaussian

target_val_ratio = 1 / np.e**2  # 0.135
max_target_val = int(255 / (np.e**2))  # 34


def compute_diameters(img):
    height, width = img.shape

    horizontal_diameter_ls = [0] * height
    horizontal_position_ls = [0] * (max_target_val + 1)

    vertical_diameter_ls = [0] * width
    vertical_position_ls2d = [[0] * (max_target_val + 1)] * width

    max_vertical_val_ls = [0] * width

    max_val = 0
    max_r = 0
    max_c = 0

    for r in range(height):
        max_horizontal_val = 0
        for c in range(width):
            v = img[r, c]

            # horizontal
            if v > max_horizontal_val:  # increasing
                if v < max_target_val:
                    for i in range(max_horizontal_val, v):
                        horizontal_position_ls[i] = c
                max_horizontal_val = v
            else:  # decreasing
                if v <= int(max_horizontal_val * target_val_ratio):
                    horizontal_diameter_ls[r] = c - horizontal_position_ls[v]
                    continue

            # vertical
            if v > max_vertical_val_ls[c]:  # increasing
                if v < max_target_val:
                    for i in range(max_vertical_val_ls[c], v):
                        vertical_position_ls2d[c][i] = r
                max_vertical_val_ls[c] = v
            else:  # decreasing
                if v <= int(max_vertical_val_ls[c] * target_val_ratio):
                    vertical_diameter_ls[c] = r - vertical_position_ls2d[c][v]

            # max_val
            if v > max_val:
                max_r = r
                max_c = c
                max_val = v

    summary = "\n".join(
        [
            f"horizontal: {horizontal_diameter_ls}",
            f"vertical: {vertical_diameter_ls}",
            f"max: {max_val} at {max_r, max_c}",
            f"diameters: {vertical_diameter_ls[max_r], horizontal_diameter_ls[max_c]}",
        ]
    )
    print(summary)


if __name__ == "__main__":

    height = 40
    width = 40

    img = generate_gaussian(height=height, width=width, max=200)
    compute_diameters(img)
