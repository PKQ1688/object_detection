# -*- coding:utf-8 -*-
# @author :adolf
import cv2
import json
import numpy as np
import os


def make_mask_img(img_path, gt_path, mask_path, img_name):
    # _, img_name = os.path.split(img_path)
    img = cv2.imread(os.path.join(img_path, img_name))
    with open(os.path.join(gt_path, img_name.replace('JPEG', 'json')), 'r') as f:
        gt = json.loads(f.read())
        gt = gt['shapes']

    for i in range(len(gt)):
        one_mask = gt[i]
        points = one_mask['points']
        img_mask = np.zeros((img.shape[:2]), dtype=np.uint8)
        points_list = [[int(j) for j in i] for i in points]
        area = np.array(points_list)
        cv2.fillPoly(img_mask, [area], i + 1)
        cv2.imwrite(os.path.join(mask_path, img_name), img_mask)


if __name__ == '__main__':
    img_path = 'data/gaoda/gao_complete/imgs/'
    gt_path = 'data/gaoda/gao_complete/gt/'
    mask_path = 'data/gaoda/gao_complete/masks/'
    img_list = os.listdir(img_path)
    for img_name in img_list:
        try:
            make_mask_img(img_path, gt_path, mask_path, img_name)
        except Exception as e:
            print(e)
            pass
