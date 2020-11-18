import pathlib
import xml.etree.ElementTree as ET
import cv2
from taskboard_transform import Taskboard
import sys
sys.path.append("../..")
import vision_tools as vision

# import pygame
# import pygame.camera
#
# pygame.camera.init()
# pygame.camera.list_camera() #Camera detected or not
# cam = pygame.camera.Camera("/dev/video0", (640,480))
# cam.start()
# img = cam.get_image()

taskboard = Taskboard("./taskboard_template/example_webcam.xml")

img_bgr = cv2.imread('CapturedImage.jpg')
# img_bgr = vision.capture_pic(remap=True)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

objects = taskboard.get_taskboard_objects(img_rgb)

for key in objects.keys():
    print("object in position {} is {}".format(objects[key][0], key))
