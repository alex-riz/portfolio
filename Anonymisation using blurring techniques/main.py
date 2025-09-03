import cv2
import numpy as np
import matplotlib
import blurring_methods as b
import request as r

matplotlib.use("TkAgg")

if __name__ == "__main__":
    input_image = cv2.imread('image.jpg')

    methods = {
        b.average_blur: np.zeros(10),
        b.median_blur: np.zeros(10),
        b.gaussian_blur: np.zeros(10),
        b.bilateral_filter: np.zeros(10)
    }
    kernel_sizes = np.arange(3, 22, 2)

    for kernel_size in kernel_sizes:
        blurred = input_image
        cnt = 0
        while True:
            blurred = b.bilateral_filter(blurred, kernel_size, 35, 35)

            cnt += 1
            cv2.imwrite('blurred.png', blurred)
            is_recognized = r.recognised('blurred.png')

            if not is_recognized:
                methods[b.bilateral_filter][kernel_sizes.index(kernel_size)] = cnt
                print(cnt)
                break



