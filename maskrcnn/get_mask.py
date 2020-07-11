# -*- coding:utf-8 -*-
# @author :adolf
import cv2
import json
import numpy as np
import os
from functools import reduce
import operator
import math


def make_mask_img(img_path, gt_path, mask_path, img_name):
    # _, img_name = os.path.split(img_path)
    img = cv2.imread(os.path.join(img_path, img_name))
    # with open(os.path.join(gt_path, img_name.replace('png', 'json')), 'r') as f:
    #     gt = json.loads(f.read())
    #     gt = gt['shapes']
    gt = []
    with open(os.path.join(gt_path, img_name.replace('jpg', 'txt').replace('png', 'txt')), 'r') as f:
        for line in f.readlines():
            line = line.strip().split(',')
            gt.append(line[:-1])

    img_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(len(gt)):
        one_mask = gt[i]
        # print(one_mask)
        # points = one_mask['points']/home/shizai/data2/ocr_data/midv_500/home/shizai/data2/ocr_data/midv_500

        # print(one_mask)
        points = [int(float(i)) for i in one_mask]
        # points = np.array(points)
        # points = points.reshape((4, 2))
        # print(points)
        # print(len(points))
        points = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]
        # print(points)
        points = clockwise_points(points)
        points = np.array(points)
        # print(points)

        # points_list = [[int(j) for j in i] for i in points]
        #     area = np.array(points_list)
        #     print(area)
        #     # unet
        cv2.fillPoly(img_mask, [points], 1)
    #     # maskrcnn
    #     # cv2.fillPoly(img_mask, [area], i + 1)
    cv2.imwrite(os.path.join(mask_path, img_name), img_mask)


def txt_to_list(txt_path):
    txt_list = list()
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            one_point_list = line.strip().split(',')
            one_point_list = [int(float(i)) for i in one_point_list]
            txt_list.append(one_point_list)

    return txt_list


def make_mask_gen_img(img_path, gt_path, mask_path, img_name):
    img = cv2.imread(os.path.join(img_path, img_name))
    txt_path = os.path.join(gt_path, img_name.replace('png', 'txt'))

    gt_list = txt_to_list(txt_path)
    img_mask = np.zeros((img.shape[:2]), dtype=np.uint8)
    area = np.array(gt_list, dtype=np.int32)
    # print(area)
    cv2.fillPoly(img_mask, [area], 1)

    cv2.imwrite(os.path.join(mask_path, img_name), img_mask)


def clockwise_points(point_coords):
    """
    以左上角为起点的顺时针排序
    原理就是讲笛卡尔坐标转换为极坐标，然后对极坐标的**进行排序
    :param point_coords: 待排序的点[(x,y)]
    :return: 排完序的点
    """
    center_point = tuple(
        map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), point_coords), [len(point_coords)] * 2))
    # for m_point in point_coords:
    #     print(m_point, math.degrees(math.atan2(*tuple(map(operator.sub, m_point, center_point))[::-1])))
    return sorted(point_coords, key=lambda coord: (135 - math.degrees(
        math.atan2(*tuple(map(operator.sub, coord, center_point))[::-1]))) % 360)


if __name__ == '__main__':
    from tqdm import tqdm
    img_path = '/home/shizai/data2/ocr_data/third_data/imgs/'
    gt_path = '/home/shizai/data2/ocr_data/third_data/gts/'
    mask_path = '/home/shizai/data2/ocr_data/third_data/masks/'
    img_list = os.listdir(img_path)
    for img_name in tqdm(img_list):
        # print(img_name)
        try:
            make_mask_img(img_path, gt_path, mask_path, img_name)
        except Exception as e:
            print(img_name)
            print(e)
            pass
        # break
    # img_path = '/home/shizai/datadisk2/ocr_data/idcard_detection/imgs/'
    # gt_path = '/home/shizai/datadisk2/ocr_data/idcard_detection/gts/'
    #
    # mask_path = '/home/shizai/datadisk2/ocr_data/idcard_detection/masks/'
    # img_list = os.listdir(img_path)
    # for img_name in img_list:
    #     try:
    #         make_mask_gen_img(img_path, gt_path, mask_path, img_name)
    #     except Exception as e:
    #         print(e)

    # img = cv2.imread(os.path.join(mask_path, 'res_2757.png'))
    # img = 100 * img
    # cv2.imwrite(os.path.join(mask_path, 'res_2757_.png'), img)
