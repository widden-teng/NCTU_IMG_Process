import numpy as np
import cv2
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from PIL import Image
from random import randint
import math
import random


if __name__ == "__main__":

    channel_initials = list('RGB')
    image = cv2.imread('./LovePeace rose.tif')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    kernel = np.ones((3, 3)) * (-1)
    kernel[1][1] = 9
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    (H, S, V) = cv2.split(hsv)
    v_sharp = cv2.filter2D(src=V, ddepth=-1, kernel=kernel)
    hsv_sharpen = np.dstack((H, S, v_sharp))
    img_sharpen = cv2.filter2D(image, -1, kernel)
    after_hrv_sharpen = cv2.cvtColor(hsv_sharpen, cv2.COLOR_HSV2BGR)

    # RGB unsharpen
    plt.subplot(221)
    plt.imshow(image, cmap='gray')
    for channel_index in range(3):
        BGR_channel = np.zeros(shape=image.shape, dtype=np.uint8)
        BGR_channel[:, :, channel_index] = image[:, :, channel_index]
        z = 222+channel_index
        plt.subplot(z)
        plt.imshow(BGR_channel, cmap='gray')
        PILimage = Image.fromarray(BGR_channel.astype(np.uint8))
        PILimage.save(
            "img/(b)" + str(channel_initials[channel_index]) + ".png", dpi=(200, 200))
    plt.show()

    # unsharpen hsv
    plt.subplot(221)
    plt.imshow(H, cmap='gray')
    plt.subplot(222)
    plt.imshow(S, cmap='gray')
    plt.subplot(223)
    plt.imshow(V, cmap='gray')
    plt.show()
    PILimage = Image.fromarray(H.astype(np.uint8))
    PILimage.save("img/(c)H.png", dpi=(200, 200))
    PILimage = Image.fromarray(S.astype(np.uint8))
    PILimage.save("img/(c)S.png", dpi=(200, 200))
    PILimage = Image.fromarray(V.astype(np.uint8))
    PILimage.save("img/(c)V.png", dpi=(200, 200))

    # RGB sharpen
    plt.subplot(221)
    plt.imshow(img_sharpen, cmap='gray')
    for channel_index in range(3):
        BGR_channel = np.zeros(shape=img_sharpen.shape, dtype=np.uint8)
        BGR_channel[:, :, channel_index] = img_sharpen[:, :, channel_index]
        z = 222+channel_index
        plt.subplot(z)
        plt.imshow(BGR_channel, cmap='gray')
        PILimage = Image.fromarray(BGR_channel.astype(np.uint8))
        PILimage.save(
            "img/(b)sharpen_" + str(channel_initials[channel_index]) + ".png", dpi=(200, 200))
    plt.show()
    PILimage = Image.fromarray(img_sharpen.astype(np.uint8))
    PILimage.save(
        "img/sharpen_rgb.png", dpi=(200, 200))
    # sharpen HSV
    (sharpen_H, sharpen_S, sharpen_V) = cv2.split(hsv_sharpen)
    plt.subplot(221)
    plt.imshow(sharpen_H, cmap='gray')
    plt.subplot(222)
    plt.imshow(sharpen_S, cmap='gray')
    plt.subplot(223)
    plt.imshow(sharpen_V, cmap='gray')
    plt.show()
    PILimage = Image.fromarray(sharpen_H.astype(np.uint8))
    PILimage.save("img/(c)sharpen_H.png", dpi=(200, 200))
    PILimage = Image.fromarray(sharpen_S.astype(np.uint8))
    PILimage.save("img/(c)sharpen_S.png", dpi=(200, 200))
    PILimage = Image.fromarray(sharpen_V.astype(np.uint8))
    PILimage.save("img/(c)sharpen_V.png", dpi=(200, 200))
    hsv_sharpen = cv2.cvtColor(hsv_sharpen, cv2.COLOR_BGR2RGB)
    PILimage = Image.fromarray(hsv_sharpen.astype(np.uint8))
    PILimage.save(
        "img/sharpen_hsv.png", dpi=(200, 200))

    diff1 = cv2.subtract(after_hrv_sharpen, img_sharpen)
    diff2 = cv2.subtract(img_sharpen, after_hrv_sharpen)

    diff1 = cv2.cvtColor(diff1, cv2.COLOR_BGR2RGB)
    diff2 = cv2.cvtColor(diff2, cv2.COLOR_BGR2RGB)
    diff_save = Image.fromarray(diff1.astype(np.uint8))
    diff_save.save("img/diff1.png", dpi=(200, 200))
    PILimage = Image.fromarray(diff2.astype(np.uint8))
    PILimage.save("img/difference2.png", dpi=(200, 200))
