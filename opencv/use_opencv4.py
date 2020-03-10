# -*- coding:utf-8 -*-
# @author :adolf
import cv2
import numpy as np

img = cv2.imread("data/gaoda/gao_complete/imgs/IMG_20191119_152848.JPEG")
cv2.imwrite('opencv/ori_img.png', img)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像转化为灰度图像

lap = cv2.Laplacian(img, cv2.CV_64F)  # 拉普拉斯边缘检测
lap = np.uint8(np.absolute(lap))  ##对lap去绝对值

contours, hierarchy = cv2.findContours(lap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

newImg = np.zeros_like(img)
newImg.fill(255)

# 画图
cv2.drawContours(newImg, contours, -1, (0, 0, 0), 3)
cv2.imwrite("opencv/test.png", newImg)
