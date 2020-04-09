# -*- coding:utf-8 -*-
# @author :adolf
import os
import paddlehub as hub

humanseg = hub.Module(name='deeplabv3p_xception65_humanseg')

path = 'data/test/'

files = [path + i for i in os.listdir(path)]

results = humanseg.segmentation(data={'image': files})
