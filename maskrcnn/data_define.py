# -*- coding:utf-8 -*-
# @author :adolf

import os
import sys

import numpy as np
import torch
from PIL import Image
import torch.utils.data as data
import detection_util.utils as utils


class OcrUseDataset_maskrcnn(data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms

        self.imgs = list(sorted(os.listdir(os.path.join(root, "img"))))
        self.gts = list(sorted(os.listdir(os.path.join(root, "gt"))))

        self.imgs = [img for img in self.imgs if 'rctw' in img]
        # self.imgs = self.imgs[index_a:]

        self.gts = [gt for gt in self.gts if 'rctw' in gt]

        # self.gts = self.gts[index_a:]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "img", self.imgs[idx])
        label_path = os.path.join(self.root, "gt", self.gts[idx])

        img = Image.open(img_path).convert("RGB")
        # gt = np.loadtxt(label_path, delimiter=',', dtype='str')
        w, h = img.size
        lines = [line.split(',') for line in open(label_path)]
        gt = [[float(x.replace('\ufeff', '')) for x in line[:8]] for line in lines]
        # print(gt)
        num_objs = len(gt)

        # masks = np.zeros((w, h), dtype=np.uint8)
        mask = np.loadtxt('/home/shizai/adolf/ai+rpa/ocr/ocr_pra/maskrcnn/dataset/mask_rctw_gt/' + self.gts[idx])

        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        # print(obj_ids)
        masks = mask == obj_ids[:, None, None]

        boxes = []
        for i in range(num_objs):
            ploy = gt[i]
            x_min = min(int(ploy[0]), int(ploy[6]))
            x_max = max(int(ploy[2]), int(ploy[4]))
            y_min = min(int(ploy[1]), int(ploy[3]))
            y_max = max(int(ploy[5]), int(ploy[7]))
            boxes.append([x_min, y_min, x_max, y_max])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.from_numpy(np.array(masks, dtype=np.uint8))

        image_id = torch.tensor([idx])

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        is_crowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = dict()

        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        # target["area"] = area
        target["iscrowd"] = is_crowd

        # print('1111111', masks.shape)
        # print('2222222', masks)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, self.imgs[idx]

    def print_img_name(self, it_):
        print(self.imgs[it_])


if __name__ == '__main__':
    # dataset = PennFudanDataset('PennFudanPed', transforms=None)  # , get_transform(train=True))
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)
    # collate_fn=utils.collate_fn)
    # print('test')
    dataset = OcrUseDataset('/home/shizai/datadisk2/ocr_data/train')
    # print(dataset.imgs)
    # dataset.__getitem__(2)
    # print(dataset.__len__())
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                              shuffle=False, num_workers=4,
                                              collate_fn=utils.collate_fn)
    # data_set = data_loader.dataset.print_img_name(3)

    img_data_dict = data_loader.dataset.__dict__
    # print(img_data_dict)
    img_data_dataset = img_data_dict['dataset']
    # print(img_data_dataset.__dict__)
    img_data_dataset_dict = img_data_dataset.__dict__

    img_list = img_data_dataset_dict['imgs']
    print(img_list[100])
