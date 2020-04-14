# -*- coding:utf-8 -*-
# @author :adolf
import os
import chardet

folder = "/home/shizai/datadisk2/ocr_data/idcard_detection/jsons/"
folder_new = "/home/shizai/datadisk2/ocr_data/idcard_detection/jsons_new/"
if not os.path.exists(folder_new):
    os.mkdir(folder_new)

listDir = os.listdir(folder)
print(listDir)
for json_name in listDir:
    with open(os.path.join(folder, json_name), encoding='gb18030') as fb:
        content = fb.read()
    with open(os.path.join(folder_new, json_name), 'w', encoding='utf-8') as ff:
        ff.write(content)
