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

import pickle

import cv2
import numpy as np

# def get_transform(train=True):
#     transforms_l = list()
#     transforms_l.append(T.ToTensor())
#     if train:
#         transforms_l.append(T.RandomHorizontalFlip(0.5))
#     return T.Compose(transforms_l)


transform1 = transforms.Compose([transforms.ToTensor()])


def one_img_predict_mask(img_path):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    img_ = Image.open(img_path).convert("RGB")
    img = transform1(img_)

    model = get_model_instance_segmentation(num_classes=2)

    model.to(device)
    # model.load_state_dict(torch.load('model_use/gen_20000.pth'))
    weights_path = '/home/shizai/data2/model/maskrcnn/model_final.pkl'
    with open(weights_path, 'rb') as f:
        obj = f.read()
    a = pickle.loads(obj, encoding='latin1')['blobs']
    del a['weight_order']
    weights = {}
    for key, arr in a.items():
        # print(key)
        # print(arr)
        # break
        weights[key] = torch.from_numpy(arr)
    # sys.exit(1)
    # weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
    model.load_state_dict(weights)

    model.eval()
    with torch.no_grad():
        outputs = model([img.to(device)])
    img1 = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    # img2 = Image.fromarray(outputs[0]['masks'][0, 0].mul(255).byte().cpu().numpy())
    img1.save('test_data/img1.png')

    image1 = cv2.cvtColor(np.asanyarray(img1), cv2.COLOR_RGB2BGR)
    OLD_IMG = image1.copy()

    # re_masks = outputs['masks']
    # print(outputs)
    result_roi = torchvision.ops.nms(outputs[0]['boxes'], outputs[0]['scores'], iou_threshold=0.5)
    print(result_roi)
    result_roi_list = result_roi.tolist()

    boxes = outputs[0]['boxes']
    scores = outputs[0]['scores']

    num = boxes.shape[0]

    for i in range(num):
        # print(scores[i].item())
        if scores[i].item() > 0.5 and i in result_roi_list:
            cv2.rectangle(image1, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (0, 0, 255), 3)
            # cv2.putText(image1, str(scores[i].item()), (boxes[i][0], boxes[i][1] - 10), cv2.FONT_HERSHEY_COMPLEX,
            #             10, (0, 0, 255), 25)

    cv2.imwrite('test_data/img_box.png', image1)

    # mask = np.zeros(image1.shape[:2], np.uint8)
    # SIZE = (1, 65)
    # bgdModle = np.zeros(SIZE, np.float64)
    # fgdModle = np.zeros(SIZE, np.float64)
    # rect = (boxes[i][0], boxes[i][1], (boxes[i][2] - boxes[i][0]), (boxes[i][3] - boxes[i][1]))
    # cv2.grabCut(image1, mask, rect, bgdModle, fgdModle, 20, cv2.GC_INIT_WITH_RECT)
    #
    # mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    # image1 *= mask2[:, :, np.newaxis]
    #
    # cv2.imwrite('test_data/img_cut' + str(i) + '.png', image1)
    #
    # break

    # mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    # img *= mask2[:, :, np.newaxis]


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    img_path = "test_data/test_10.png"
    one_img_predict_mask(img_path)
