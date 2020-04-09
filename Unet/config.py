# -*- coding:utf-8 -*-
# @author :adolf

batch_size = 16
epochs = 100
lr = 0.0001,

device = "cuda:0",
workers = 4,

vis_images = 200  # number of visualization images to save in log file (default: 200)
vis_freq = 10,  # frequency of saving images to log file (default: 10)

weights = "./weights"  # folder to save weights
logs = "./logs"  # folder to save logs
images = "./data"  # root folder with images

image_size = 256  # target input image size (default: 256)

aug_scale = 0.05  # scale factor range for augmentation (default: 0.05)
aug_angle = 15  # rotation angle range in degrees for augmentation (default: 15)
