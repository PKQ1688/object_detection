# -*- coding:utf-8 -*-
# @author :adolf
import detection_util.transforms as T
from maskrcnn.data_define import *
import torch.utils.data
from maskrcnn.model_define import *
import os
import torchvision
from PIL import Image
import torchvision.transforms as transforms

import cv2
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_transform(train=True):
    transforms_l = list()
    transforms_l.append(T.ToTensor())
    if train:
        transforms_l.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms_l)


transform1 = transforms.Compose([transforms.ToTensor()])


def one_img_predict_mask(img_path, is_gts=False):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    img_ = Image.open(img_path).convert("RGB")
    img = transform1(img_)

    img_name = os.path.basename(img_path)
    # pre_name = img_name.split('.')[0] + '.txt'
    model = get_model_instance_segmentation(num_classes=2)

    model.to(device)
    model.load_state_dict(torch.load('model_use/gen_20000.pth'))

    model.eval()
    with torch.no_grad():
        outputs = model([img.to(device)])
    outputs = outputs  # [0]
    # print(outputs)
    # re_masks = outputs['masks']
    img1 = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    # img2 = Image.fromarray(outputs[0]['masks'][0, 0].mul(255).byte().cpu().numpy())
    img1.save('test_data/img1.png')

    image1 = cv2.cvtColor(np.asanyarray(img1), cv2.COLOR_RGB2BGR)

    for i in range(len(outputs)):
        img = Image.fromarray(outputs[i]['masks'][0, 0].mul(255).byte().cpu().numpy())
        img.save('test_data/mask_' + str(i) + '.png')

        image = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # image, contours, hierarchy = cv2.findContours(thresh, 3, 2)
        # cnt = contours[0]

        # 2.进行多边形逼近，得到多边形的角点
        # approx = cv2.approxPolyDP(cnt, 3, True)
        #
        # print(approx)
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # cv2.polylines(image, [approx], True, (0, 255, 0), 2)

        # cv2.imwrite('test_data/img_mask_' + str(i) + '.png', image)

        ret, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(binary, 3, 2)

        cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]

        print(cnt)
        print('---------------')
        # 2.进行多边形逼近，得到多边形的角点
        hull = cv2.convexHull(cnt, 3, True)
        #
        print(hull)
        # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.polylines(image1, [hull], True, (0, 0, 255), 3)
        #
        cv2.imwrite('test_data/img_mask_' + str(i) + '.png', image1)
        # # print(contours)
        #
        # c = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        # rect = cv2.minAreaRect(c)
        # box = np.int0(cv2.boxPoints(rect))
        # print(box)
        #
        # cv2.drawContours(image1, contours, -1, (0, 0, 255), 3)
        #
        # cv2.imwrite('test_data/img_mask_' + str(i) + '.png', image1)

    # print(re_masks)
    # print(re_masks.size())
    # mask_img = re_masks.cpu().numpy()
    # print(np.sum(mask_img[0]))
    # result_roi = torchvision.ops.nms(outputs['boxes'], outputs['scores'],
    #                                  iou_threshold=0.5)
    # # print(result_roi)
    # result_roi_list = result_roi.tolist()
    # img = cv2.cvtColor(np.asarray(img_), cv2.COLOR_RGB2BGR)
    # # print(result_roi)
    #
    # boxes = outputs['boxes']
    # scores = outputs['scores']
    #
    # num = boxes.shape[0]
    # # print(num)
    # for i in range(num):
    #     # print(scores[i].item())
    #     if scores[i].item() > 0.5 and i in result_roi_list:
    #         # cv2.rectangle(img, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (0, 0, 255), 2)
    #         # cv2.putText(img, str(scores[i].item()),
    #         # (boxes[i][0], boxes[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0, 0, 255), 0.1)
    #         file_path = '/home/shizai/adolf/ai+rpa/ocr/ocr_pra/maskrcnn/dataset/mask_pre/' + pre_name
    #         with open(file_path, 'a', encoding='utf-8')as f:
    #             f.write(str(int(boxes[i][0])))
    #             f.write(',')
    #             f.write(str(int(boxes[i][1])))
    #             f.write(',')
    #             f.write(str(int(boxes[i][2])))
    #             f.write(',')
    #             f.write(str(int(boxes[i][3])))
    #             f.write(',')
    #             f.write(str(scores[i]))
    #             f.write('\n')
    #
    # if is_gts:
    #     label_path = img_path.replace('/img/', '/gt/').replace('icdar15_img_', 'icdar15_gt_img_').replace('jpg', 'txt')
    #     lines = [line.split(',') for line in open(label_path)]
    #     gts = [[float(x.replace('\ufeff', '')) for x in line[:8]] for line in lines]
    #
    #     for j in range(len(gts)):
    #         # box = quad.strip().split(',')[0:8]
    #         box = gts[j]
    #         box = np.array(box, dtype=np.float)
    #         box = np.array([[box[0], box[1]], [box[2], box[3]], [box[4], box[5]], [box[6], box[7]]], np.int32)
    #
    #         box = box.reshape((-1, 1, 2))
    #         # print(box)
    #         # print(box.shape)
    #         cv2.polylines(img, [box], True, (0, 255, 0), 2)

    # cv2.imwrite('mask_test.png', img)


if __name__ == '__main__':
    img_path = "test_data/gaoda/gao_complete/imgs/IMG_20191119_152917.JPEG"
    one_img_predict_mask(img_path, is_gts=False)
