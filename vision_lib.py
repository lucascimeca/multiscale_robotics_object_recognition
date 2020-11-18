import numpy as np
import cv2
import matplotlib.pyplot as plt


# DEBUG FUNCTION - returns plot of rgb, grayscale, hsv and lab color space channels
def show_colorspaces(img_rgb):
    f, axarr = plt.subplots(6, 3, figsize=(5, 15))
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_yuv =  cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
    img_hls =  cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)
    img_luv =  cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LUV)

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

    axarr[3, 0].imshow(img_yuv[:, :, 0], cmap='gray')
    axarr[3, 0].set_title('YUV- Y channel')
    axarr[3, 1].imshow(img_yuv[:, :, 1], cmap='gray')
    axarr[3, 1].set_title('YUV- U channel')
    axarr[3, 2].imshow(img_yuv[:, :, 2], cmap='gray')
    axarr[3, 2].set_title('YUV- V channel')

    axarr[4, 0].imshow(img_hls[:, :, 0], cmap='gray')
    axarr[4, 0].set_title('HLS- H channel')
    axarr[4, 1].imshow(img_hls[:, :, 1], cmap='gray')
    axarr[4, 1].set_title('HLS- L channel')
    axarr[4, 2].imshow(img_hls[:, :, 2], cmap='gray')
    axarr[4, 2].set_title('HLS- S channel')

    axarr[5, 0].imshow(img_luv[:, :, 0], cmap='gray')
    axarr[5, 0].set_title('LUV- L channel')
    axarr[5, 1].imshow(img_luv[:, :, 1], cmap='gray')
    axarr[5, 1].set_title('LUV- U channel')
    axarr[5, 2].imshow(img_luv[:, :, 2], cmap='gray')
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
def find_motor(image_lab):
    # threshold on typical motor lab values
    bw_image_lab1 = np.logical_and(image_lab[:, :, 2] > 133, image_lab[:, :, 2] < 150)
    bw_image_lab2 = np.logical_and(image_lab[:, :, 1] > 120, image_lab[:, :, 1] < 130)
    bw_image = np.logical_and(bw_image_lab1, bw_image_lab2).astype(np.uint8) * 255

    centers_kernel = np.ones((15, 15), np.uint8)
    closed_img = cv2.morphologyEx(bw_image, cv2.MORPH_CLOSE,
                                  centers_kernel)

    # test blobs, and retrieve only the ones that look like motors (size + circularity)
    roi_copy = image_lab.copy()
    im2, cnts, hierarchy = cv2.findContours(closed_img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)  # get largest five contour area
    for cnt in cnts:
        perimeter = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)
        if perimeter==0:
            break
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if area > 5000 and area < 9000 and circularity > 0.4 and circularity < 1.2:
            x, y, w, h = cv2.boundingRect(cnt)
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
        img = cv2.cvtColor(motor_rgb, cv2.COLOR_RGB2GRAY)
        img = cv2.GaussianBlur(img, (3, 3), 0)

        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 100, param2=15, minRadius=np.int(img.shape[0] / 10),
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

# find circles in binary image
def detect_circles(binary_image):

    min_radius = np.min(binary_image.shape)/4
    max_radius = min_radius*2

    for radius in np.arange(min_radius, max_radius, (max_radius-min_radius)/10):

        # boundary positions of the circle center to look at
        radius = np.int32(radius)
        min_x = min_y = radius
        max_x = binary_image.shape[1]-radius
        max_y = binary_image.shape[0]-radius

        # #stride to apply
        # sweeps = 10  # how many sweeps over each size of the image
        # stride_top = np.int32((top_right[1] - top_left[1]) / sweeps)
        # stride_side = np.int32((bottom_left[0] - top_left[0]) / sweeps)

        for cx in np.arange(min_x, max_x, 10):
            cx = np.int32(cx)
            for cy in np.arrange(min_y, max_y):
                pass
    return

# find motor orientation from rgb image
def find_motor_orientation(image_rgb):
    roi = None
    theta = None
    image_copy_rgb = image_rgb.copy()
    image_lab = cv2.cvtColor(image_copy_rgb, cv2.COLOR_RGB2LAB)
    roi, motor_loc = find_motor(image_lab)
    if motor_loc is not None:
        x, y, w, h = motor_loc  # unpack
        cropped_motor = image_copy_rgb[y:y + h, x:x + w]

        # find shaft
        motor_pos = np.array([np.int32(x + w / 2), np.int32(y + h / 2)])
        shaft_loc = find_motor_shaft(cropped_motor, mode='circles')
        if shaft_loc is not None:
            shaft_pos = shaft_loc + np.array([x, y])

            # compute angle from motor and shaft location
            theta = np.rad2deg(np.arctan2(*(shaft_pos[::-1] - motor_pos[::-1])))

            # draw in roi for display
            roi = cv2.cvtColor(roi, cv2.COLOR_LAB2RGB)
            compass_vect = np.int32((shaft_pos - motor_pos) * 5 + motor_pos)
            cv2.line(roi, tuple(motor_pos), tuple(compass_vect), (255, 0, 0), 15)
            cv2.putText(roi, str(theta), (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(255, 0, 0), thickness=3)

            return roi, theta
    return image_rgb, None


def bound_contours(roi, mask):
    """
        returns modified roi(non-destructive) and rectangles that founded by the algorithm.
        @roi region of interest to find contours
        @return (roi, rects)
    """
    roi_copy = roi.copy()
    # Find contours for detected portion of the image
    im2, cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    rects = []
    for c in cnts:
        rects += [cv2.minAreaRect(c)]
        box = cv2.boxPoints(rects[-1])
        box = np.int0(box)
        cv2.drawContours(roi_copy, [box], 0, (0, 0, 255), 2)
    return roi_copy, rects


def crop_minAreaRect(img, rect, scale=1):
    box = cv2.boxPoints(rect)
    W = rect[1][0]
    H = rect[1][1]

    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)

    rotated = False
    angle = rect[2]

    if angle < -45:
        angle += 90
        rotated = True

    center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
    size = (int(scale * (x2 - x1)), int(scale * (y2 - y1)))
    if any(np.array(size)==0):
        return None
    # cv2.circle(img_box, center, 10, (0, 255, 0), -1)  # again this was mostly for debugging purposes

    M = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)

    cropped = cv2.getRectSubPix(img, size, center)
    cropped = cv2.warpAffine(cropped, M, size)

    croppedW = W if not rotated else H
    croppedH = H if not rotated else W

    croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW * scale), int(croppedH * scale)),
                                       (size[0] / 2, size[1] / 2))

    return croppedRotated, (x1, y2)


def template_match(image, template):
    # img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # img = cv2.GaussianBlur(img_gray, (5, 5), 0)
    cimg = image.copy()

    sigma = 0.33
    v = np.median(image)
    upper = int(min(255, (1.0 + sigma) * v))

    circles = cv2.HoughCircles(cropped_img, cv2.HOUGH_GRADIENT, 15, 100,
                               param1=upper, param2=100, minRadius=0, maxRadius=0)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    # cv2.imshow('detected circles',cimg

    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    ax = axes.flatten()

    ax[0].imshow(image, cmap='gray')
    ax[0].set_axis_off()
    # ax[0].set_title('sobel')
    ax[1].imshow(cimg, cmap='gray')
    ax[1].set_axis_off()
    # ax[1].set_title('gaussian')
    plt.show()
    plt.close(fig)


# returns padded version of cropped image
def cropped_padder(cropped_image, width=300, height=300):
    padded_img = np.zeros((height, width, cropped_image.shape[2])).astype(np.int16)

    pad_width = width - cropped_image.shape[1]
    pad_height = height - cropped_image.shape[0]

    lower_width = int(np.floor(pad_width/2))
    lower_height = int(np.floor(pad_height/2))
    upper_width = lower_width + cropped_image.shape[1]
    upper_height = lower_height + cropped_image.shape[0]

    padded_img[lower_height:upper_height, lower_width:upper_width, :] = cropped_image.copy()

    return padded_img


def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )


# load webcam image and find taskboard
def find_squares(img):
    # img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Adaptive Guassian Threshold is to detect sharp edges in the Image. For more information Google it.
    gaussian_binary = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3.5)

    kernel = np.ones((3, 3), np.uint8)
    gaussian_binary = cv2.erode((gaussian_binary == 0).astype(np.uint8) * 255, kernel, iterations=1)
    gaussian_binary = cv2.dilate(gaussian_binary, kernel, iterations=1)
    centers_kernel = np.ones((15, 15), np.uint8)
    closed_img = cv2.morphologyEx(gaussian_binary, cv2.MORPH_CLOSE, centers_kernel)

    closed_img, contours, _hierarchy = cv2.findContours(closed_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))


    for cnt in cntsSorted:
        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
        if len(cnt) == 4 and cv2.contourArea(cnt) > 1000 and cv2.isContourConvex(cnt):
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([angle_cos(cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4]) for i in range(4)])
            if max_cos < 0.1:
                squares.append(cnt)
    return squares