import cv2
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic


def pic_convert(path1):
    """
    this function chage the raw image file to the RGB version
    :param path1: the path of the image
    :return: the image in RGB version
    """
    img1 = Image.open(path1)
    pattern = 'gbrg'
    img1 = demosaic(img1, pattern)
    img1 = np.array(img1).astype(np.uint8)
    img1 = img1[:, :, [2, 1, 0]]
    return img1


# path = './ref'
path = './query'
file_name_list = os.listdir(path)
print(file_name_list)
print(file_name_list[17304])
i = 0
for item in file_name_list[17304:]:
    name = path + '/'+item
    try:
        img = pic_convert(name)
        cv2.imwrite('/media/pinoc/Nicole/slam_data/navigation_data/query/'+item, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    except Exception as e:
        print(str(e))
        print(item)
