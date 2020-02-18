import random
import torch

from torchvision.transforms import functional as F


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class ScaleResize(object):
    def __init__(self, shortest_side=600):
        self.shortest_side = shortest_side

    def __call__(self, image, target):
        width = image.size[0]
        height = image.size[1]

        # scale = float(self.shortest_side) / float(min(height, width))
        if height > width:
            scale = float(self.shortest_side) / float(width)
            image = F.resize(image, (self.shortest_side, int(height * scale)), 2)
        else:
            scale = float(self.shortest_side) / float(height)
            image = F.resize(image, (int(width * scale), self.shortest_side), 2)

        h_scale = float(image.size[1]) / float(height)
        w_scale = float(image.size[0]) / float(width)

        scale_gt = []
        # print('2target', target)
        bbox = target["boxes"]
        for box in bbox:
            scale_box = []
            for i in range(len(box)):
                if i % 2 == 0:
                    scale_box.append(int(int(box[i]) * w_scale))
                else:
                    scale_box.append(int(int(box[i]) * h_scale))
            scale_gt.append(scale_box)
        boxes = torch.as_tensor(scale_gt, dtype=torch.float32)
        target["boxes"] = boxes
        return image, target
