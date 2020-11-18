import numpy as np
import cv2
import matplotlib.pyplot as plt


# DEBUG FUNCTION - returns plot of rgb, grayscale, hsv and lab color space channels
def show_colorspaces(img_rgb):
    f, axarr = plt.subplots(6, 3, figsize=(15, 15))
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    image_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
    image_hls = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)
    image_luv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LUV)

    axarr[0, 0].imshow(img_rgb)
    axarr[0, 0].set_title('rgb')
    axarr[0, 1].imshow(img_gray, cmap='gray')
    axarr[0, 1].set_title('grayscale')

    axarr[1, 0].imshow(img_hsv[:, :, 0], cmap='gray')
    axarr[1, 0].set_title('HSV- H channel')
    axarr[1, 1].imshow(img_hsv[:, :, 1], cmap='gray')
    axarr[1, 1].set_title('HSV- S channel')
    axarr[1, 2].imshow(img_hsv[:, :, 2], cmap='gray')
    axarr[1, 2].set_title('HSV- V channel')

    axarr[2, 0].imshow(img_lab[:, :, 0], cmap='gray')
    axarr[2, 0].set_title('LAB- L channel')
    axarr[2, 1].imshow(img_lab[:, :, 1], cmap='gray')
    axarr[2, 1].set_title('LAB- A channel')
    axarr[2, 2].imshow(img_lab[:, :, 2], cmap='gray')
    axarr[2, 2].set_title('LAB- B channel')

    axarr[3, 0].imshow(image_yuv[:, :, 0], cmap='gray')
    axarr[3, 0].set_title('YUV- Y channel')
    axarr[3, 1].imshow(image_yuv[:, :, 1], cmap='gray')
    axarr[3, 1].set_title('YUV- U channel')
    axarr[3, 2].imshow(image_yuv[:, :, 2], cmap='gray')
    axarr[3, 2].set_title('YUV- V channel')

    axarr[4, 0].imshow(image_hls[:, :, 0], cmap='gray')
    axarr[4, 0].set_title('HLS- H channel')
    axarr[4, 1].imshow(image_hls[:, :, 1], cmap='gray')
    axarr[4, 1].set_title('HLS- L channel')
    axarr[4, 2].imshow(image_hls[:, :, 2], cmap='gray')
    axarr[4, 2].set_title('HLS- S channel')

    axarr[5, 0].imshow(image_luv[:, :, 0], cmap='gray')
    axarr[5, 0].set_title('LUV- L channel')
    axarr[5, 1].imshow(image_luv[:, :, 1], cmap='gray')
    axarr[5, 1].set_title('LUV- U channel')
    axarr[5, 2].imshow(image_luv[:, :, 2], cmap='gray')
    axarr[5, 2].set_title('LUV- V channel')
    
    return f, axarr


# given an image and a bounding rectangle, crops and returns the area within the rectangle
def crop_min_area_rect(img, rect, scale=1):
    box = cv2.boxPoints(rect)
    w = rect[1][0]
    h = rect[1][1]

    xs = [i[0] for i in box]
    ys = [i[1] for i in box]
    x1 = min(xs)
    x2 = max(xs)
    y1 = min(ys)
    y2 = max(ys)

    rotated = False
    angle = rect[2]
    if angle < -45:
        angle += 90
        rotated = True

    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    size = (int(scale * (x2 - x1)), int(scale * (y2 - y1)))
    if any(np.array(size) == 0):
        return None
    m = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)

    cropped = cv2.getRectSubPix(img, size, center)
    cropped = cv2.warpAffine(cropped, m, size)
    croppedW = w if not rotated else h
    croppedH = h if not rotated else w
    cropped_rotated = cv2.getRectSubPix(cropped, (int(croppedW * scale), int(croppedH * scale)),
                                       (size[0] / 2, size[1] / 2))
    return cropped_rotated


# Function which given a camera image, returns the location of the motor within.
def find_motor(image_rgb):

    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    img_yuv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YUV)
    # threshold on typical motor lab values
    bw_image1 = np.logical_and(image_lab[:, :, 2] > 137, image_lab[:, :, 2] < 145)
    bw_image2 = np.logical_and(img_yuv[:, :, 1] > 113, img_yuv[:, :, 1] < 122)
    bw_image = np.logical_and(bw_image1, bw_image2).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    closed_img = cv2.morphologyEx(bw_image, cv2.MORPH_ERODE,
                                  kernel, iterations=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_img = cv2.morphologyEx(closed_img, cv2.MORPH_CLOSE,
                                  kernel, iterations=5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_img = cv2.morphologyEx(closed_img, cv2.MORPH_ERODE,
                                  kernel, iterations=5)

    # test blobs, and retrieve only the ones that look like motors (size + circularity)
    roi_copy = image_lab.copy()
    im2, cnts, hierarchy = cv2.findContours(closed_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)  # get largest five contour area
    for cnt in cnts:
        perimeter = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        if perimeter == 0:
            break
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if area > 5000 and area < 14000 and circularity > 0.35 and circularity < 1.2:
            x, y, w, h = cv2.boundingRect(cnt)
            if x > 900 and x < 1100 and y > 300 and y < 550:
                cv2.rectangle(roi_copy, (x, y), (x + w, y + h), (0, 255, 0), 4)
                return roi_copy, (x, y, w, h)
    return roi_copy, None


# Function which given a motor image, returns the location of the shaft within.
def find_motor_shaft(motor_rgb, mode=None):
    if mode is None:
        motor_lab = cv2.cvtColor(motor_rgb, cv2.COLOR_RGB2LAB)

        bw_image = np.logical_and(motor_lab[:, :, 2] > 146, motor_lab[:, :, 2] < 153).astype(np.uint8) * 255

        centers_kernel = np.ones((3, 3), np.uint8)
        closed_img = cv2.morphologyEx(bw_image, cv2.MORPH_ERODE,
                                      centers_kernel)
        centers_kernel = np.ones((5, 5), np.uint8)
        closed_img = cv2.morphologyEx(closed_img, cv2.MORPH_CLOSE,
                                      centers_kernel)

        roi_copy = motor_lab.copy()
        im2, cnts, hierarchy = cv2.findContours(closed_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)  # get largest five contour area
        for cnt in cnts:
            perimeter = cv2.arcLength(cnt, True)
            area = cv2.contourArea(cnt)
            if perimeter == 0:
                break
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if area > 20 and area < 250 and circularity > 0.35 and circularity < 1.2:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(roi_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                return np.array([np.int32(x+w/2), np.int32(y+h/2)])
        return None

    elif mode == 'template_match':
        img = cv2.cvtColor(motor_rgb, cv2.COLOR_RGB2GRAY)
        template = cv2.imread('./templates/motor_axis.jpg', 0)
        template = cv2.resize(template, (np.int32(img.shape[0] / 6), np.int32(img.shape[0] / 6)))
        w, h = template.shape[::-1]

        # Apply template Matching
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        cx, cy = np.int32(max_loc[0] + w / 2), np.int32(max_loc[1] + h / 2)
        return np.array([cx, cy])

    elif mode == 'circles':
        img = cv2.cvtColor(motor_rgb, cv2.COLOR_RGB2LAB)[:, :, 0]
        img = cv2.GaussianBlur(img, (3, 3), 0)

        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 2, 100,
                                   param1=200,
                                   param2=30,
                                   minRadius=np.int(img.shape[0] / 8),
                                   maxRadius=np.int(img.shape[0] / 5))
        if circles is not None:

            # --- DEBUG ---
            # cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            # circles = np.array(np.uint16(np.around(circles)))
            # for i in circles[0, :]:
            #     # draw the outer circle
            #     cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            #     # draw the center of the circle
            #     cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

            circles = circles[0][0]
            return circles[:2]
        else:
            return None
    else:
        raise ValueError('supply a supported mode')
