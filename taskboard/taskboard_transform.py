import sys
import numpy as np
import cv2
import pathlib
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
# sys.path.append("..")
# from vision_lib import find_square
# from vision.vision_lib import find_squares



class Taskboard():

    objects = None
    classes = None

    def __init__(self, path_to_template):
        self.taskboard = None
        self.path_to_template = path_to_template
        self.template_obj_dict = dict()
        self.__load()

    def __load(self):
        # file = pathlib.Path(self.path_to_template)

        # get xml file and find drawn boxes
        tree = ET.parse(self.path_to_template)
        root = tree.getroot()

        objs = root.findall('object')
        for ix, obj in enumerate(objs):
            # get object info
            bndbox = obj.find('bndbox')
            obj_class = obj.find('name').text
            # frame in figure
            x1 = int(bndbox[0].text)
            y1 = int(bndbox[1].text)
            x2 = int(bndbox[2].text)
            y2 = int(bndbox[3].text)
            self.template_obj_dict[obj_class] = [x1, y1, x2, y2]


    # returns a dictionary where each key is an object name['2', '3' ... etc],
    # and the value is it's location in the taskboard
    def get_taskboard_objects(self, taskboard_rgb):
        obj_dict = dict()

        #------ FIND TASKBOARD AND DO AFFINE TRANSFORAMTION -------#

        # find taskboard (returns corners as array in order [up-left, up-right, bottom-right, bottom-left])
        # taskboard = find_squares(taskboard_rgb)[0]
        taskboard = [[594, 125], [1274, 121], [1270, 800], [593, 796]]
        x_min = np.min([taskboard[0][0], taskboard[3][0]])
        x_max = np.max([taskboard[1][0], taskboard[2][0]])
        y_min = np.min([taskboard[0][1], taskboard[1][1]])
        y_max = np.max([taskboard[3][1], taskboard[2][1]])

        # roi = taskboard_rgb.copy()
        # for pos in taskboard:
        #     cv2.circle(roi, tuple(pos), 5, (255, 0, 0))
        # plt.imshow(roi)
        # plt.show()

        # affine transformation
        brd = self.template_obj_dict['board']  # x1, y1, x2, y2
        pts1 = np.array([[brd[0], brd[1]], [brd[2], brd[1]], [brd[2], brd[3]], [brd[0], brd[3]]]).astype(np.float32)
        pts2 = np.array(taskboard).astype(np.float32)
        M = cv2.getPerspectiveTransform(pts1, pts2)

        roi = taskboard_rgb.copy()

        # guess object position given warp
        taskboard_obj_dict = dict()
        for key in self.template_obj_dict.keys():
            x1, y1, x2, y2 = self.template_obj_dict[key]
            obj_center = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]]).astype(np.float32)
            warped_center = cv2.perspectiveTransform(np.array([obj_center]), M, (obj_center.shape[1], obj_center.shape[0]))
            taskboard_obj_dict[key] = warped_center[0][0]

            cv2.putText(roi, key, tuple(taskboard_obj_dict[key]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        color=(255, 0, 0), thickness=3)

        self.classes = np.array([cls.strip() for cls in taskboard_obj_dict.keys() if cls.strip().isdigit()])# array holding which class obj belongs to
        self.objects = np.zeros((self.classes.shape[0], 2))# array holding centers of objects

        for i in range(len(self.objects)):
            self.objects[i, :] = taskboard_obj_dict[self.classes[i]]

        # create taskboard mask for cleaning
        offset = 20
        taskboard_mask = np.zeros((taskboard_rgb.shape[0], taskboard_rgb.shape[1]))
        taskboard_mask[y_min+offset:y_max-offset, x_min+offset:x_max-offset] = 1

        # --------- FIND ALL OBJECTS IN IMAGE -------------#
        img_gray = cv2.cvtColor(taskboard_rgb, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

        # Adaptive Guassian Threshold is to detect sharp edges in the Image. For more information Google it.
        gaussian_binary = cv2.adaptiveThreshold(img_blur,
                                                255,
                                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY,
                                                15,
                                                3.5)

        kernel = np.ones((3, 3), np.uint8)
        gaussian_binary = cv2.erode((gaussian_binary == 0).astype(np.uint8) * 255, kernel, iterations=1)
        gaussian_binary = cv2.dilate(gaussian_binary, kernel, iterations=1)
        centers_kernel = np.ones((15, 15), np.uint8)
        closed_img = cv2.morphologyEx(gaussian_binary, cv2.MORPH_CLOSE, centers_kernel)
        closed_img = np.logical_and(closed_img == 255, taskboard_mask==1).astype(np.uint8)*255

        #binary image
        _, cnts, hirarchy = cv2.findContours(closed_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        # tmp = taskboard_rgb.copy()
        # cv2.circle(tmp, (cX, cY), 20, (255, 0, 0))
        # cv2.drawContours(tmp, cnts, -1, (0, 255, 0), 3)
        # plt.imshow(tmp)
        # plt.show()

        # for each object guess clss given warped dictionary
        for cnt in cnts:
            M = cv2.moments(cnt)
            try:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # tmp = taskboard_rgb.copy()
                # # cv2.circle(tmp, (cX, cY), 20, (255, 0, 0))
                # cv2.drawContours(tmp, [cnt], -1, (0, 255, 0), 3)
                # plt.imshow(tmp)
                # plt.show()

                if cX > x_min and cX < x_max and cY > y_min and cY < y_max:
                    distances = list(map(lambda pt: np.sqrt((pt[0]-cX)**2 + (pt[1]-cY)**2), self.objects))
                    obj = self.classes[np.argmin(distances)]
                    if obj in obj_dict.keys():
                        obj_dict[obj] = obj_dict[obj] + [(cX, cY)]
                    else:
                        obj_dict[obj] = [(cX, cY)]
            except:
                pass


        # for cls in obj_dict.keys():
        #     cv2.putText(roi, cls, obj_dict[cls][0], cv2.FONT_HERSHEY_SIMPLEX, 1,
        #                 color=(255, 0, 0), thickness=3)
            # cv2.circle(roi, obj_dict[cls][0], 15, (0, 0, 255))

        fig, axes = plt.subplots(1, 2, figsize=(15, 15))
        ax = axes.flatten()

        ax[0].imshow(roi)
        ax[0].set_axis_off()
        ax[1].imshow(closed_img, cmap='gray')
        ax[1].set_axis_off()
        plt.show()
        plt.close(fig)

        return obj_dict


