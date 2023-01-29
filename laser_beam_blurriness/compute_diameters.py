import numpy as np
from generate_gaussian import generate_gaussian

target_val_ratio = 1 / np.e**2  # 0.135
max_target_val = int(255 / (np.e**2)) + 1  # 35


def compute_diameters(img):
    height, width = img.shape

    horizontal_diameter_ls = [0] * height

    vertical_diameter_ls = [0] * width
    vertical_position_ls2d = [([0] * (max_target_val + 1)) for _ in range(width)]
    max_vertical_val_ls = [0] * width

    max_val = 0
    max_r = 0
    max_c = 0

    for r in range(height):
        horizontal_position_ls = [0] * (max_target_val + 1)
        max_horizontal_val = 0
        for c in range(width):
            v = img[r, c]
            clipped_v = min(v, max_target_val)

            # horizontal
            if v > max_horizontal_val:  # increasing
                if max_horizontal_val < clipped_v:
                    for i in range(max_horizontal_val + 1, clipped_v + 1):
                        assert horizontal_position_ls[i] == 0
                        horizontal_position_ls[i] = c
                max_horizontal_val = v
            else:  # decreasing
                if (v <= int(max_horizontal_val * target_val_ratio)) and (
                    horizontal_diameter_ls[r] == 0
                ):
                    horizontal_diameter_ls[r] = c - horizontal_position_ls[v]

            # vertical
            if v > max_vertical_val_ls[c]:  # increasing
                if max_vertical_val_ls[c] < clipped_v:
                    for i in range(max_vertical_val_ls[c] + 1, clipped_v + 1):
                        assert vertical_position_ls2d[c][i] == 0
                        vertical_position_ls2d[c][i] = r
                max_vertical_val_ls[c] = v
            else:  # decreasing
                if (v <= int(max_vertical_val_ls[c] * target_val_ratio)) and (
                    vertical_diameter_ls[c] == 0
                ):
                    vertical_diameter_ls[c] = r - vertical_position_ls2d[c][v]

            # max_val
            if v > max_val:
                max_r = r
                max_c = c
                max_val = v

    summary = "\n".join(
        [
            f"horizontal_position_ls: {horizontal_position_ls}",
            f"vertical_position_ls: {vertical_position_ls2d}",
            f"horizontal_diameter_ls: {horizontal_diameter_ls}",
            f"vertical_diameter_ls: {vertical_diameter_ls}",
            f"max_val: {max_val} at {max_r, max_c}",
            f"diameters at max_val point : {vertical_diameter_ls[max_r], horizontal_diameter_ls[max_c]}",
        ]
    )
    print(summary)


if __name__ == "__main__":

    height = 10
    width = 10

    img = generate_gaussian(height=height, width=width, max=200)
    compute_diameters(img)
