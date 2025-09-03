import numpy as np


def average_blur(image, kernel_size):
    height, width, channels = image.shape
    blurred_image = np.zeros_like(image, dtype=np.uint8)
    padding = kernel_size // 2

    for y in range(padding, height - padding):
        for x in range(padding, width - padding):
            for c in range(channels):
                blurred_image[y, x, c] = np.mean(image[y - padding:y + padding + 1, x - padding:x + padding + 1, c])

    return blurred_image


def gaussian_blur(image, kernel_size, sigma):
    height, width, channels = image.shape
    blurred_image = np.zeros_like(image, dtype=np.uint8)
    padding = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)

    for y in range(-padding, padding + 1):
        for x in range(-padding, padding + 1):
            kernel[y + padding, x + padding] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    kernel /= 2 * np.pi * sigma ** 2

    for y in range(padding, height - padding):
        for x in range(padding, width - padding):
            for c in range(channels):
                blurred_image[y, x, c] = np.sum(
                    image[y - padding:y + padding + 1, x - padding:x + padding + 1, c] * kernel)

    return blurred_image


def median_blur(image, kernel_size):
    height, width, channels = image.shape
    blurred_image = np.zeros_like(image, dtype=np.uint8)
    padding = kernel_size // 2

    for y in range(padding, height - padding):
        for x in range(padding, width - padding):
            for c in range(channels):
                blurred_image[y, x, c] = np.median(image[y - padding:y + padding + 1, x - padding:x + padding + 1, c])

    return blurred_image


def bilateral_filter(image, diameter, sigma_color, sigma_space):
    height, width, channels = image.shape
    blurred_image = np.zeros_like(image, dtype=np.uint8)
    padding = diameter // 2

    for y in range(padding, height - padding):
        for x in range(padding, width - padding):
            bilateral = np.zeros_like(image[y, x], dtype=np.float64)
            total_weight = 0.0

            for i in range(-padding, padding + 1):
                for j in range(-padding, padding + 1):
                    if (y + i >= 0 and y + i < height and x + j >= 0 and x + j < width):
                        spatial_dist = np.exp(-((i ** 2 + j ** 2) / (2 * sigma_space ** 2)))
                        color_dist = np.exp(
                            -(((image[y, x] - image[y + i, x + j]) ** 2).sum() / (2 * sigma_color ** 2)))
                        weight = spatial_dist * color_dist
                        bilateral += image[y + i, x + j] * weight
                        total_weight += weight

            blurred_image[y, x] = (bilateral / total_weight).astype(np.uint8)

    return blurred_image



