
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow
import argparse
from argparse import ArgumentParser


path = "/content/drive/MyDrive/ImageProcessing/Assignment-24"


parser = ArgumentParser()
parser.add_argument("--stiker", type=str)
parser.add_argument("--image", type=str)
parser.add_argument("--stiker_eye", type=str)
parser.add_argument("--stiker_mouth", type=str)

# parser.add_argument("--OutPut_FaceStikerName", type=str)
# parser.add_argument("--OutPut_SmileStikerName", type=str)
# parser.add_argument("--OutPut_EyeStikerName", type=str)

args = parser.parse_args()


face_detector_file = cv2.CascadeClassifier(os.path.join(path+ "/haarcascade_frontalface_default.xml"))
smile_detector_file = cv2.CascadeClassifier(os.path.join(path+ "/haarcascade_smile.xml"))
eye_detector_file = cv2.CascadeClassifier(os.path.join(path+ "/haarcascade_eye.xml"))

image = cv2.imread(args.image)
image_gray_face_stikr = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_gray_mouth_eye_stiker = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_gray_blur = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

stiker = cv2.imread(args.stiker, cv2.IMREAD_UNCHANGED)
stiker_smile = cv2.imread(args.stiker_mouth, cv2.IMREAD_UNCHANGED)
stiker_eye = cv2.imread(args.stiker_eye, cv2.IMREAD_UNCHANGED)

stikers = []
stikers.append(stiker)
stikers.append(stiker_smile)
stikers.append(stiker_eye)


def Sticker_placement(image_face, stiker, detected_shape):

      x, y, w, h = detected_shape
      stiker_image = stiker[:, :, 0:3]
      stiker_image_gray = cv2.cvtColor(stiker_image, cv2.COLOR_BGR2GRAY)
      stiker_mask = stiker[:, :, 3]

      stiker_image_gray_resize = cv2.resize(stiker_image_gray, (w, h))
      stiker_mask_resize = cv2.resize(stiker_mask, (w, h))

      stiker_image_gray_resize = stiker_image_gray_resize.astype(float) / 255
      stiker_mask_resize = stiker_mask_resize.astype(float) / 255

      image_face = image_face.astype(float) / 255

      forgraound = cv2.multiply(stiker_image_gray_resize, stiker_mask_resize)
      background = cv2.multiply(image_face, 1 - stiker_mask_resize)
      
      result = cv2.add(forgraound, background)
      result = result * 255

      return result



faces = face_detector_file.detectMultiScale(image_gray_face_stikr, 1.3)

for face in faces:

    x, y, w, h = face

    image_face = image_gray_face_stikr[y : y+h, x : x+w]
    stiker = stikers[0]
    result = Sticker_placement(image_face, stiker, face)
    image_gray_face_stikr[y : y+h, x : x+w] = result



    image_faces = image_gray_mouth_eye_stiker[y : y+h, x : x+w]
    image_eye = eye_detector_file.detectMultiScale(image_faces, 1.3)

    for eye in image_eye:

        x_e, y_e, w_e, h_e = eye
        image_smile = image_faces[y_e : y_e+h_e, x_e : x_e+w_e]
        result = Sticker_placement(image_smile, stikers[2], eye)
        image_faces[y_e : y_e+h_e, x_e : x_e+w_e] = result


    image_smile = smile_detector_file.detectMultiScale(image_faces, 1.4)

    for smile in image_smile:

        x_s, y_s, w_s, h_s = smile
        image_smile = image_faces[y_s : y_s+h_s, x_s : x_s+w_s]
        result = Sticker_placement(image_smile, stikers[1], smile)
        image_faces[y_s : y_s+h_s, x_s : x_s+w_s] = result


    image_gray_mouth_eye_stiker[y : y+h, x : x+w] = image_faces

#-----blur

    new_w = w//8
    new_h = h//8

    image_face = image_gray_blur[y : y+h, x : x+w]

    image_face_resize1 = cv2.resize(image_face, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    image_face_resize2 = cv2.resize(image_face_resize1, (w, h), interpolation=cv2.INTER_NEAREST)

    image_gray_blur[y : y+h, x : x+w] = image_face_resize2


cv2.imwrite(os.path.join(path, "Result/image_gray_face_stikr.jpg"), image_gray_face_stikr)
cv2.imwrite(os.path.join(path, "Result/image_gray_mouth_eye_stiker.jpg"), image_gray_mouth_eye_stiker)
cv2.imwrite(os.path.join(path, "Result/image_gray_blur.jpg"), image_gray_blur)

cv2_imshow(image_gray_face_stikr)
cv2_imshow(image_gray_mouth_eye_stiker)
cv2_imshow(image_gray_blur)