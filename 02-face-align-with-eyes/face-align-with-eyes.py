
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import argparse
import os
from mtcnn.mtcnn import MTCNN
import matplotlib.pyplot as plt
from PIL import Image
import math
from functions import *
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--input_image", type=str)
args = parser.parse_args()

path = "/content/drive/MyDrive/ImageProcessing/Assignment-27"

img = cv2.imread(args.input_image)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#--------------------------face-align-with-eyes

face_detector = MTCNN()

# draw an image with detected objects


faces = face_detector.detect_faces(img)

if faces==[]:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_detector.detect_faces(img)

draw_facebox_and_keypoints("/content/drive/MyDrive/ImageProcessing/Assignment-27/inputs/mr_bean.jpeg", faces)


#-------------------------

detection = faces[0]
keypoints = detection["keypoints"]
left_eye = keypoints["left_eye"]
right_eye = keypoints["right_eye"]

img = alignment_procedure(img, left_eye, right_eye)

cv2.imwrite(os.path.join(path, "result/face-align-with-eyes_mr_bean.jpg"), img)
plt.imshow(img)