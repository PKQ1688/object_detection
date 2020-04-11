# -*- coding:utf-8 -*-
# @author :adolf
import os
import random

import numpy as np
import torch
from PIL import Image

from torch.utils.data import Dataset

from skimage.io import imread

from utils import crop_sample, pad_sample, resize_sample, normalize_volume
from transforms import get_transforms


class UNetSegmentationDataset(Dataset):
    in_channels = 3
    out_channels = 1

    def __init__(
            self,
            images_dir,
            image_size=256,
            transform=None,
            random_sampling=True,
            subset="train",
    ):
        self.images_dir = images_dir
        self.transform = transform
        self.image_size = image_size
        self.random_sampling = random_sampling

        assert subset in ["all", "train", "validation"]

        volumes = {}
        masks = {}

        # print("begining")

        img_list = os.listdir(os.path.join(self.images_dir, "masks"))
        # print(img_list)
        img_list = img_list[:30]
        for img_name in img_list:
            # print(img_name)
            mask_slices = [imread(os.path.join(self.images_dir, "masks", img_name), as_gray=True)]
            image_slices = [imread(os.path.join(self.images_dir, "imgs", img_name))]

            volumes[img_name] = np.array(image_slices)
            masks[img_name] = np.array(mask_slices)

        # print('load image is end .....')
        self.patients = sorted(volumes)

        # validation_cases = 10
        validation_cases = int(0.1 * len(self.patients))

        if not subset == "all":
            validation_patients = random.sample(self.patients, k=validation_cases)
            if subset == "validation":
                self.patients = validation_patients
            else:
                self.patients = sorted(
                    list(set(self.patients).difference(validation_patients))
                )

        # print("preprocessing {} volumes...".format(subset))
        self.volumes = [(volumes[k], masks[k]) for k in self.patients]

        # print("cropping {} volumes...".format(subset))
        self.volumes = [crop_sample(v) for v in self.volumes]

        # print("padding {} volumes...".format(subset))
        self.volumes = [pad_sample(v) for v in self.volumes]

        # print("resizing {} volumes...".format(subset))
        self.volumes = [resize_sample(v, size=image_size) for v in self.volumes]

        # print("normalizing {} volumes...".format(subset))
        import time
        s1 = time.time()
        self.volumes = [(normalize_volume(v), m) for v, m in self.volumes]
        # print('cost time', time.time() - s1)

        self.volumes = [(v, m[..., np.newaxis]) for (v, m) in self.volumes]

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, idx):
        # print(self.volumes)
        image, mask = self.volumes[idx]

        # v, m = self.volumes[]
        image = image[0]
        mask = mask[0]

        if self.transform is not None:
            image, mask = self.transform((image, mask))

        # fix dimensions (C, H, W)
        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        image_tensor = torch.from_numpy(image.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))

        # return tensors
        return image_tensor, mask_tensor


if __name__ == '__main__':
    train_datasets = UNetSegmentationDataset(
        images_dir="/data3/adolf/ocr_hup/object_detection/data/gaoda/12_08",
        image_size=256,
        subset="train",
        transform=get_transforms(scale=0.05, angle=15, flip_prob=0.5),
    )
    train_datasets.__getitem__(2)
