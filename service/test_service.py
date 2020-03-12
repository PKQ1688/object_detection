# -*- coding:utf-8 -*-
# @author :adolf
import cv2
import requests
import json
import base64
import numpy as np

from PIL import Image, ImageDraw, ImageFont


def get_result(encodestr):
    payload = {"image": encodestr, "type": "image"}
    r = requests.post("http://192.168.1.135:1314/profile_service/", json=payload)
    # print(r.text)
    res = json.loads(r.text)
    # print(res)
    return res


def put_text(img, text, left, top):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype("仿宋_GB2312.ttf", 30, encoding="utf-8")
    draw.text((left, top - 30), text, (255, 0, 0), font=fontText)
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    return img


def one_image(img_path):
    import os
    o_img = cv2.imread(img_path)

    with open(img_path, 'rb') as f:
        image = f.read()
        encodestr = str(base64.b64encode(image), 'utf-8')

    res_ = get_result(encodestr)
    return res_


if __name__ == '__main__':
    import time

    s1 = time.time()
    # while True:
    file_path = "test_data/gaoda/gao_complete/imgs/IMG_20191120_140856.JPEG"
    res = one_image(file_path)
    print(res)
