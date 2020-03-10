# -*- coding:utf-8 -*-
# @author :adolf
import detection_util.transforms as T
from maskrcnn.data_define import *
import torch.utils.data
from maskrcnn.model_define import *
import os
from PIL import Image
import torchvision.transforms as transforms

import cv2
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "4"


def get_transform(train=True):
    transforms_l = list()
    transforms_l.append(T.ToTensor())
    if train:
        transforms_l.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms_l)


transform1 = transforms.Compose([transforms.ToTensor()])

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = get_model_instance_segmentation(num_classes=2)

model.to(device)
model.load_state_dict(torch.load('model_use/gen_20000.pth'))


def predict(img):
    img = transform1(img)

    model.eval()
    with torch.no_grad():
        outputs = model([img.to(device)])
    outputs = outputs

    print(outputs)
    print('box num:', len(outputs))
    # img1 = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())

    box_dict = dict()

    for i in range(len(outputs)):
        img = Image.fromarray(outputs[i]['masks'][0, 0].mul(255).byte().cpu().numpy())
        img.save('data/mask_' + str(i) + '.png')

        image = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print(contours)

        cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        # rect = cv2.minAreaRect(c)
        # box = np.int0(cv2.boxPoints(rect))
        hull = cv2.convexHull(cnt, 3, True)
        #
        # print(hull)

        box_dict[i] = hull

    return box_dict
