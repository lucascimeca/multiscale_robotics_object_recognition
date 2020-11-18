import pathlib
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt
import numpy as np


# path to imagenet dataset
PATH_TO_DATA = 'data/imagenet_data/'
WHICH_SET = 'train_set/'
IMG_FORMAT = 'jpg'
PATH_TO_SAVE = 'data/tf_data/'


# returns padded version of cropped image
def cropped_padder(cropped_image, width=700, height=500):
    padded_img = np.zeros((height, width, cropped_image.shape[2])).astype(np.int16)

    pad_width = width - cropped_image.shape[1]
    pad_height = height - cropped_image.shape[0]

    lower_width = int(np.floor(pad_width/2))
    lower_height = int(np.floor(pad_height/2))
    upper_width = lower_width + cropped_image.shape[1]
    upper_height = lower_height + cropped_image.shape[0]

    padded_img[lower_height:upper_height, lower_width:upper_width, :] = cropped_image.copy()

    return padded_img


# CONVERSION
i = 0
for file in pathlib.Path(PATH_TO_DATA + WHICH_SET).glob('**/*.txt'):
    path_to_file = str(file)
    objects = [line for line in file.read_text().split('\n') if len(line) != 0]

    num_objs = len(objects)
    for ix, obj_line in enumerate(objects):
        # get object info
        bndbox = obj_line.split(" ")[1:]
        obj_class = int(obj_line.split(" ")[0])

        # create object data folder if not there
        pathlib.Path(PATH_TO_SAVE + WHICH_SET + '{}'.format(obj_class)).mkdir(parents=True, exist_ok=True)

        path = path_to_file.replace('labels_yolo', 'images').replace('txt', IMG_FORMAT)
        img = cv2.imread(path, 1)
        w = img.shape[1]
        h = img.shape[0]

        bbox_width = float(bndbox[2]) * w
        bbox_height = float(bndbox[3]) * h

        center_x = float(bndbox[0]) * w
        center_y = float(bndbox[1]) * h
        x1 = int(np.floor(center_x - (bbox_width / 2)))
        y1 = int(np.floor(center_y - (bbox_height / 2)))
        x2 = int(np.floor(center_x + (bbox_width / 2)))
        y2 = int(np.floor(center_y + (bbox_height / 2)))

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cropped_image = cropped_padder(img[y1:y2, x1:x2, :], width=640, height=640)

        # DEBUG
        print('found object {} in {}. Processed {}'.format(obj_class, str(bndbox), i))

        # fig, axes = plt.subplots(1, 2, figsize=(15, 15))
        # ax = axes.flatten()
        # ax[0].imshow(img)
        # ax[0].set_axis_off()
        # ax[0].set_title('original')
        # ax[1].imshow(cropped_image)
        # ax[1].set_axis_off()
        # ax[1].set_title('cropped, class {}'.format(str(obj_class)))
        # plt.show()

        cv2.imwrite(PATH_TO_SAVE + WHICH_SET + '{}/image{}.{}'.format(obj_class, i, IMG_FORMAT), cropped_image)

        i+=1


