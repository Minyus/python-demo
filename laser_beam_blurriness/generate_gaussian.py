import numpy as np
import cv2 as cv


def generate_gaussian(height=200, width=200, max=255):
    kh = height - 1
    kw = width - 1
    assert kh // 2
    assert kw // 2
    assert max <= 255

    img = np.zeros((height, width))
    img[height // 2, width // 2] = 1
    img = cv.GaussianBlur(img, (kw, kh), 0)
    img = ((img / img.max()) * max).astype(np.uint8)
    return img


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    img = generate_gaussian(max=200)

    plt.imshow(img)
    plt.show()
