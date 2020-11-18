import pathlib
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt
import numpy as np


# path to imagenet dataset
PATH_TO_DATA = 'data/imagenet_data/'
WHICH_SET = 'valid_set/'
IMG_FORMAT = 'jpg'
PATH_TO_SAVE = 'data/tf_data/'


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


# CONVERSION
i = 0
for file in pathlib.Path(PATH_TO_DATA + WHICH_SET).glob('**/*.xml'):
    path_to_file = str(file)

    # get xml files from PATH_TO_DATA
    tree = ET.parse(file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    objs = root.findall('object')
    num_objs = len(objs)
    for ix, obj in enumerate(objs):
        # get object info
        bndbox = obj.find('bndbox')
        obj_class = obj.find('name').text

        # create object data folder if not there
        pathlib.Path(PATH_TO_SAVE + WHICH_SET + '{}'.format(obj_class)).mkdir(parents=True, exist_ok=True)

        # cut out frame in figure
        x1 = int(bndbox[0].text)
        y1 = int(bndbox[1].text)
        x2 = int(bndbox[2].text)
        y2 = int(bndbox[3].text)

        path = path_to_file.replace('labels', 'images').replace('xml', IMG_FORMAT)
        img = cv2.imread(path, 1)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cropped_image = cropped_padder(img[y1:y2, x1:x2, :])


        # DEBUG
        print('found object {} in {}. Processed {}'.format(obj_class, str(bndbox), i))

        # fig, axes = plt.subplots(1, 2, figsize=(15, 15))
        # ax = axes.flatten()
        # ax[0].imshow(img)
        # ax[0].set_axis_off()
        # ax[0].set_title('original')
        # ax[1].imshow(cropped_image)
        # ax[1].set_axis_off()
        # ax[1].set_title('cropped, class {}'.format(str(obj_class+4)))
        # plt.show()

        cv2.imwrite(PATH_TO_SAVE + WHICH_SET + '{}/image{}.{}'.format(obj_class, i, IMG_FORMAT), cropped_image)

        i+=1


