# -*- coding:utf-8 -*-
# @author :adolf
import cv2
import numpy as np

img = cv2.imread("data/gaoda/gao_complete/imgs/IMG_20191119_152848.JPEG")
cv2.imwrite('opencv/ori_img.png', img)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像转化为灰度图像

sobelX = cv2.Sobel(img, cv2.CV_64F, 1, 0)  # x方向的梯度
sobelY = cv2.Sobel(img, cv2.CV_64F, 0, 1)  # y方向的梯度

sobelX = np.uint8(np.absolute(sobelX))  # x方向梯度的绝对值
sobelY = np.uint8(np.absolute(sobelY))  # y方向梯度的绝对值

sobelCombined = cv2.bitwise_or(sobelX, sobelY)

cv2.imwrite('opencv/sobel.png', sobelCombined)

contours, hierarchy = cv2.findContours(sobelCombined, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

newImg = np.zeros_like(img)
newImg.fill(255)

# 画图
cv2.drawContours(newImg, contours, -1, (0, 0, 0), 3)
cv2.imwrite("opencv/test.png", newImg)
