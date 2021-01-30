import math
import os

import numpy as np
from cv2 import cv2
from PIL import Image, ImageDraw

dataset_path = "./train/s41"

mouseclick = []


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    global mouseclick
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseclick.append((x, y))
        print(x, y)
        if len(mouseclick) == 2:
            print("finish!")
    return


def myeyelocate(img_path):
    global mouseclick
    output_path = img_path[0:-4] + ".txt"
    with open(output_path, 'w+') as file:
        if file.read() != "":
            file.close()
            return
        img = cv2.imread(img_path)
        cv2.namedWindow("eyes")
        cv2.setMouseCallback("eyes", on_EVENT_LBUTTONDOWN)
        cv2.imshow("eyes", img)
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()

        if len(mouseclick) >= 2:
            x1, y1 = mouseclick[0]
            x2, y2 = mouseclick[1]
            file.write(str(x1) + ' ' + str(y1) + ' ' + str(x2) + ' ' + str(y2))
            mouseclick.clear()

        file.close()
    return


def mydistance(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx*dx+dy*dy)


def ScaleRotateTranslate(image, angle, center=None, new_center=None, scale=None, resample=Image.BICUBIC):
    if (scale is None) and (center is None):
        return image.rotate(angle=angle, resample=resample)
    nx, ny = x, y = center
    sx = sy = 1.0
    if new_center:
        (nx, ny) = new_center
    if scale:
        (sx, sy) = (scale, scale)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    a = cosine/sx
    b = sine/sx
    c = x-nx*a-ny*b
    d = -sine/sy
    e = cosine/sy
    f = y-nx*d-ny*e
    return image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f), resample=resample)


def CropFace(image, eye_left=(0, 0), eye_right=(0, 0), offset_pct=(0.2, 0.2), dest_sz=(70, 70)):
    # calculate offsets in original image
    offset_h = math.floor(float(offset_pct[0])*dest_sz[0])
    offset_v = math.floor(float(offset_pct[1])*dest_sz[1])
    # get the direction
    eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
    # calc rotation angle in radians
    rotation = -math.atan2(float(eye_direction[1]), float(eye_direction[0]))
    # distance between them
    dist = mydistance(eye_left, eye_right)
    # calculate the reference eye-width
    reference = dest_sz[0] - 2.0*offset_h
    # scale factor
    scale = float(dist)/float(reference)
    # rotate original around the left eye
    image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)
    # crop the rotated image
    crop_xy = (eye_left[0] - scale*offset_h, eye_left[1] - scale*offset_v)
    crop_size = (dest_sz[0]*scale, dest_sz[1]*scale)
    image = image.crop((int(crop_xy[0]), int(crop_xy[1]), int(
        crop_xy[0]+crop_size[0]), int(crop_xy[1]+crop_size[1])))
    # resize it
    image = image.resize(dest_sz, Image.ANTIALIAS)
    return image


if __name__ == "__main__":
    # 选眼睛点
    for i in range(1, 11):
        face_path = f'{dataset_path}/{i}.jpg'
        myeyelocate(face_path)
        eyes_path = f'{dataset_path}/{i}.txt'
        with open(eyes_path) as file:
            [x1, y1, x2, y2] = file.read().split(' ')
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            img = Image.open(face_path)
            CropFace(img, eye_left=(x1, y1), eye_right=(x2, y2), offset_pct=(
                0.3, 0.3)).save(f'{dataset_path}/masked_{i}.jpg')
