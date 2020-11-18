import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from wrs_recongition_cnn import WrsRecognitionCNN
from vision_lib import (bound_contours,
                        crop_minAreaRect,
                        cropped_padder,
                        show_colorspaces)

# LOAD AND CONVERT

wrsCnn = WrsRecognitionCNN('recognition_CNN/networks/gray_net.pb')

# Morphological ACWE
# img_bgr = cv2.imread('./../CroppedUndistorted_PlacementMat.jpg', 1)
img_bgr = cv2.imread('./templates/example2.jpg', 1)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

# remove noise
# img = cv2.GaussianBlur(img_gray, (5, 5), 0)

# SOBEL DETECTION

# convolute with proper kernels
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)  # x
sobely = cv2.Sobel(img,cv2.CV_64F, 0, 1, ksize=5)  # y

thresh = 1000
sobelx_ext = np.logical_or(sobelx < -thresh, sobelx > thresh)
sobely_ext = np.logical_or(sobely < -thresh, sobely > thresh)
sobel_binary = np.logical_or(sobely_ext, sobelx_ext)

# ADAPTIVE GAUSSIAN DETECTION

output = img_rgb.copy()
gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
# gray = cv2.medianBlur(gray,5)

# Adaptive Guassian Threshold is to detect sharp edges in the Image. For more information Google it.
gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3.5)

kernel = np.ones((3, 3), np.uint8)
gray = cv2.erode(gray, kernel, iterations=1)
gray = cv2.dilate(gray, kernel, iterations=1)

gaussian_binary = gray.copy() == 0


# circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2, 20,
#                            minRadius=3,
#                            maxRadius=np.int32(gaussian_binary.shape[0]*(1/3)))
# cimg = draw_circles(gray, circles)

centers_kernel = np.ones((15, 15), np.uint8)
closed_img = cv2.morphologyEx(np.logical_or(gaussian_binary, sobel_binary).astype(np.uint8)*255, cv2.MORPH_CLOSE, centers_kernel)

roi, rects = bound_contours(img_rgb, closed_img)

input_batch = np.zeros((len(rects), wrsCnn.height, wrsCnn.width, wrsCnn.channels))
coords = np.zeros((len(rects), 2))

j = 0
for rect in rects:
    # crop part
    res = crop_minAreaRect(img_gray, rect)  # cropped image
    if res is not None:
        cropped_img, (x, y) = res
        if cropped_img is not None and cropped_img.shape[0] < 300 and cropped_img.shape[1] < 300:
            input_batch[j, :, :, :] = cropped_padder(cropped_img.reshape((cropped_img.shape[0], cropped_img.shape[1], 1)))
            coords[j, :] = [x, y]
            j += 1

batch_size = j
classes, predictions = wrsCnn.predict(input_batch[:j, :, :])
for i in range(batch_size):
    cv2.putText(roi, str(classes[i]), tuple(coords[i, :].astype(np.int32)), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 0, 0), thickness=2)


############### PLOT #############
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
ax = axes.flatten()

ax[0].imshow(sobel_binary, cmap='gray')
ax[0].set_axis_off()
ax[0].set_title('sobel')
ax[1].imshow(gaussian_binary, cmap='gray')
ax[1].set_axis_off()
ax[1].set_title('gaussian')
ax[2].imshow(closed_img, cmap='gray')
ax[2].set_axis_off()
ax[2].set_title('gaussian or sobel')
ax[3].imshow(roi)
ax[3].set_axis_off()
ax[3].set_title('circles')
plt.show()
