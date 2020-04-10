# -*- coding:utf-8 -*-
# @author :adolf
import os
import random

import numpy as np
import torch
from PIL import Image

from torch.utils.data import Dataset

# from utils import crop_sample, pad_sample, resize_sample, normalize_volume


class UNetSegmentationDataset(Dataset):
    in_channels = 3
    out_channels = 1

    def __init__(
            self,
            images_dir,
            image_size=256,
            transform=None,
            random_sampling=True,
    ):
        self.images_dir = images_dir
        self.transform = transform
        self.image_size = image_size
        self.random_sampling = random_sampling

        self.masks = list(sorted(os.listdir(os.path.join(images_dir, "masks"))))

        # self.patients = sorted(volumes)

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, "imgs", self.masks[idx])
        mask_path = os.path.join(self.images_dir, "masks", self.masks[idx])

        img = Image.open(img_path).convert("RGB")
        img = np.array(img)

        mask = Image.open(mask_path)
        mask = np.array(mask)

        if self.transform is not None:
            img, mask = self.transform((img, mask))

        img = img.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        image_tensor = torch.from_numpy(img.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))

        return image_tensor, mask_tensor
