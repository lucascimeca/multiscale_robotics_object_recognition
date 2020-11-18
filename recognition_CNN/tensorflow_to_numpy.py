import pathlib
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
random.seed(20123)

image_limit = 2000

# path to imagenet dataset
PATH_TO_DATA = 'data/tf_data/'
WHICH_SET = 'train_set/'
IMG_FORMAT = 'jpg'
PATH_TO_SAVE = 'data/np_data/'
OUT_FORMAT = 'lab'

def read_img(path_to_file):
    if OUT_FORMAT == 'gray':
        return cv2.cvtColor(cv2.imread(path_to_file, 1), cv2.COLOR_BGR2GRAY).flatten()
    elif OUT_FORMAT == 'jpg':
        return cv2.cvtColor(cv2.imread(path_to_file, 1), cv2.COLOR_BGR2RGB).flatten()
    elif OUT_FORMAT == 'lab':
        return cv2.cvtColor(cv2.imread(path_to_file, 1), cv2.COLOR_BGR2LAB).flatten()

file_list = list(pathlib.Path(PATH_TO_DATA + WHICH_SET).glob('**/*.jpg'))
random.shuffle(file_list)
data_size = len(list(pathlib.Path(PATH_TO_DATA + WHICH_SET).glob('**/*.jpg')))

exfile = file_list[0]
path_to_file = str(exfile)
img = read_img(path_to_file)

data_splits = list(np.arange(0, data_size, image_limit))
data_splits += [data_splits[-1] + data_size % image_limit]
print("Deviding the data in {} batches of {} images".format(data_splits, image_limit))

data_size = image_limit
data = np.zeros((data_size, img.shape[0]))

label_map = {}

for i in range(1, len(data_splits)):
    # CONVERSION
    labels = np.zeros((data_size))
    j = 0
    for file in file_list:
        path_to_file = str(file)
        data[j, :] = read_img(path_to_file)
        img_class = path_to_file.split('\\')[3]
        labels[j] = np.int32(img_class)
        if labels[j] not in label_map.keys():
            label_map[labels[j]] = img_class

         print('processed image number {}'.format(j))
        j += 1
        if j >= image_limit:
            break

    print('saving...')
    pathlib.Path(PATH_TO_SAVE + WHICH_SET).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PATH_TO_SAVE + WHICH_SET + 'data_{}'.format(i),
                        inputs=data,
                        targets=labels,
                        idx_limit=j,
                        label_map=label_map,
                        format=OUT_FORMAT,
                        num_classes=5)
print('done')

