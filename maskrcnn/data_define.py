# -*- coding:utf-8 -*-
# @author :adolf

import os
import sys

import numpy as np
import torch
from PIL import Image
import torch.utils.data as data
import detection_util.utils as utils


class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class mask_use_data(data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms

        self.masks = list(sorted(os.listdir(os.path.join(root, "12_08_mask"))))

        ignore_list = ['IMG_20191118_140303_1.JPEG', 'IMG_20191119_150755_1.JPEG',
                       'IMG_20191119_143046.JPEG', 'IMG_20191119_144209.JPEG',
                       'IMG_20191119_151026.JPEG', 'IMG_20191118_144541.JPEG',
                       'IMG_20191118_140229.JPEG', 'IMG_20191119_135621.JPEG',
                       'IMG_20191119_135635.JPEG', 'IMG_20191118_164210.JPEG ']
        self.masks = [img for img in self.masks if img not in ignore_list]
        self.imgs = self.masks

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "12_08_img", self.masks[idx])
        mask_path = os.path.join(self.root, "12_08_mask", self.masks[idx])
        img = Image.open(img_path).convert("RGB")

        # img = Image.fromarray(np.uint8(img))

        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        masks = mask == obj_ids[:, None, None]

        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


if __name__ == '__main__':
    # dataset = PennFudanDataset('PennFudanPed', transforms=None)  # , get_transform(train=True))
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)
    # collate_fn=utils.collate_fn)
    # print('test')
    dataset = mask_use_data('/home/shizai/datadisk2/ocr_data/train')
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
