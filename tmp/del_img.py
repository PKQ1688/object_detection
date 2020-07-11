# -*- coding:utf-8 -*-
# @author :adolf
import cv2
import os
from tqdm import tqdm

img_path = '/home/shizai/data2/ocr_data/third_data/imgs/'

img_list = os.listdir(img_path)


def is_valid_jpg(jpg_file):
    if jpg_file.split('.')[-1].lower() == 'jpg':
        with open(jpg_file, 'rb') as f:
            f.seek(-2, 2)
            return f.read() == '\xff\xd9'
    else:
        return True


# for img_name in tqdm(img_list):
#     img = cv2.imread(os.path.join(img_path, img_name))
#     # if not is_valid_jpg(os.path.join(img_path, img_name)):
#     #     print(img_name)
#     if img.shape:
#         continue
#     else:
#         print(img_name)

img2 = cv2.imread(os.path.join(img_path, img_list[712]))
print(img_list[712])
print('--------------')
img3 = cv2.imread(os.path.join(img_path, img_list[928]))
print(img_list[928])
print('--------------')
img4 = cv2.imread(os.path.join(img_path, img_list[940]))
print(img_list[940])
print('--------------')
img5 = cv2.imread(os.path.join(img_path, img_list[3479]))
print(img_list[3479])
print('--------------')
img6 = cv2.imread(os.path.join(img_path, img_list[4325]))
print(img_list[4325])
print('--------------')