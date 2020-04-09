# -*- coding:utf-8 -*-
# @author :adolf
import cv2
import numpy as np

# 读取图片
img = cv2.imread("test_data/gaoda/gao_complete/imgs/IMG_20191119_152848.JPEG")
cv2.imwrite('opencv/ori_img.png', img)
# 转灰度图片
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# 轮廓检测
contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# 新打开一个图片，我这里这张图片是一张纯白图片
# newImg = cv2.imread("E:\\font\\aaa.bmp")
# newImg = cv2.resize(newImg, (300, 300))
newImg = np.zeros_like(img)
newImg.fill(255)

# 画图
cv2.drawContours(newImg, contours, -1, (0, 0, 0), 3)

# 展示
cv2.imwrite("opencv/test.png", newImg)
