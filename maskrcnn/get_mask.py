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


def txt_to_list(txt_path):
    txt_list = list()
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            one_point_list = line.strip().split(',')
            one_point_list = [int(i) for i in one_point_list]
            txt_list.append(one_point_list)

    return txt_list


def make_mask_gen_img(img_path, gt_path, mask_path, img_name):
    img = cv2.imread(os.path.join(img_path, img_name))
    txt_path = os.path.join(gt_path, img_name.replace('png', 'txt'))

    gt_list = txt_to_list(txt_path)
    img_mask = np.zeros((img.shape[:2]), dtype=np.uint8)
    area = np.array(gt_list)
    cv2.fillPoly(img_mask, [area], 1)

    cv2.imwrite(os.path.join(mask_path, img_name), img_mask)


if __name__ == '__main__':
    # img_path = 'test_data/gaoda/gao_complete/imgs/'
    # gt_path = 'test_data/gaoda/gao_complete/gt/'
    # mask_path = 'test_data/gaoda/gao_complete/masks/'
    # img_list = os.listdir(img_pa
    # for img_name in img_list:
    #     try:
    #         make_mask_img(img_path, gt_path, mask_path, img_name)
    #     except Exception as e:
    #         print(e)
    #         pass
    img_path = '/home/shizai/data2/ocr_data/gaoda/gaoda_gen/imgs/'
    gt_path = '/home/shizai/data2/ocr_data/gaoda/gaoda_gen/gt/'
    mask_path = '/home/shizai/data2/ocr_data/gaoda/gaoda_gen/masks'
    img_list = os.listdir(img_path)
    for img_name in img_list:
        try:
            make_mask_gen_img(img_path, gt_path, mask_path, img_name)
        except Exception as e:
            print(e)

    # img = cv2.imread(os.path.join(mask_path, 'res_2757.png'))
    # img = 100 * img
    # cv2.imwrite(os.path.join(mask_path, 'res_2757_.png'), img)
