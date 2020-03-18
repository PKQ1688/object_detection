import cv2
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from object_detection.maskrcnn.model_define import get_model_instance_segmentation


class MaskRCNNPrediction(object):
    def __init__(self, params):
        self.device = params.get("device", "cpu")
        self.n_class = params.get("n_class", 2)
        self.resume = params.get("resume", False)
        self.verbose = params.get("verbose", False)

        self.transform = transforms.Compose([transforms.ToTensor()])

        self.model = get_model_instance_segmentation(num_classes=self.n_class)

        self.model.to(torch.device(self.device))

        if self.resume:
            self.model.load_state_dict(torch.load(self.resume))
            if self.verbose:
                print("load mask-rcnn model success.")
        else:
            if self.verbose:
                print("no mask-rcnn model to load")
        self.model.eval()

    def predict(self, img):
        t_img = self.transform(img)
        with torch.no_grad():
            outputs = self.model([t_img.to(torch.device(self.device))])

            result_roi = torchvision.ops.nms(outputs[0]["boxes"], outputs[0]["scores"], iou_threshold=0.5)
            result_roi_list = result_roi.tolist()

            boxes = outputs[0]["boxes"]
            scores = outputs[0]["scores"]

            num = boxes.shape[0]

            boxes_list = []
            for i in range(num):
                if scores[i].item() > 0.5 and i in result_roi_list:
                    boxes = np.array([
                        [int(boxes[i][0]), int(boxes[i][1])],
                        [int(boxes[i][2]), int(boxes[i][1])],
                        [int(boxes[i][2]), int(boxes[i][3])],
                        [int(boxes[i][0]), int(boxes[i][3])],
                    ])
                    boxes_list.append(boxes)
            return boxes_list


if __name__ == "__main__":
    params = {
        "resume": "/data1/gaoda_models/maskrcnn/gen_20000.pth"
    }
    path = "/data3/qfeng/高达软件钢牌图片已标注/高达软件钢牌图片/标签压缩/佛山市高明基业冷轧钢板有限公司/IMG_20191118_165203.JPEG"
    img = cv2.imread(path)
    model = MaskRCNNPrediction(params)
    model.predict(img)
