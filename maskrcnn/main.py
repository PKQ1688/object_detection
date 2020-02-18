# -*- coding:utf-8 -*-
# @author :adolf
import sys

sys.path.append('../')
import torch
import detection_util.transforms as T
from detection_util.engine import train_one_epoch, evaluate
import detection_util.utils as utils
from maskrcnn.data_define import *
import torch.utils.data
from maskrcnn.model_define import *
import os

# from maskrcnn.data_define import OcrUseDataset

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"


# 获取每个 GPU 的剩余显存数，并存放到 tmp 文件中
os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
os.environ["CUDA_VISIBLE_DEVICES"] = str(int(np.argmax(memory_gpu)))
# torch.cuda.set_device(np.argmax(memory_gpu))
os.system('rm tmp')  # 删除临时生成的 tmp 文件


def get_transform(train):
    transforms_l = list()
    transforms_l.append(T.ToTensor())
    if train:
        transforms_l.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms_l)


def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = "cpu"
    num_classes = 2

    # dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
    # dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))
    # dataset = OcrUseDataset(root='/home/shizai/datadisk2/ocr_data/train/',
    #                         root2='/home/shizai/datadisk2/tyin/images',
    #                         transforms=get_transform(train=True))
    # dataset_test = OcrUseDataset(root='/home/shizai/datadisk2/ocr_data/train/',
    #                              root2='/home/shizai/datadisk2/tyin/images',
    #                              transforms=get_transform(train=False))
    #
    dataset = OcrUseDataset_maskrcnn('/home/shizai/datadisk2/ocr_data/train/', get_transform(train=True))
    dataset_test = OcrUseDataset_maskrcnn('/home/shizai/datadisk2/ocr_data/train/', get_transform(train=False))

    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    print('333333333', len(dataset) + 50)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2,
                                              shuffle=False, num_workers=4,
                                              collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                   batch_size=1, shuffle=False, num_workers=4,
                                                   collate_fn=utils.collate_fn)

    model = get_model_instance_segmentation(num_classes)
    # ctpn_model.load_state_dict(torch.load('./model_use/mask_123_rcnn_model_150.pth'))
    # ctpn_model = use_model_change()
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=1e-5, momentum=0.9, weight_decay=0.0005)
    # optimizer = torch.optim.Adam(params, lr=1e-5, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 21

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        if epoch % 5 == 0:
            # evaluate(ctpn_model, data_loader_test, device=device)
            torch.save(model.state_dict(), './model_save/mask_rcnn_model_' + str(epoch) + '.pth')

    print("That's is all!")
    # torch.save(ctpn_model.state_dict(), 'mask_123_rcnn_model.pth')


if __name__ == '__main__':
    main()
    # ctpn_model = get_model_instance_segmentation(num_classes=2)
