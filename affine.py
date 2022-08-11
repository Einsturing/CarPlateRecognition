import sys
import os
import cv2
import numpy as np


numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphbets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'W', 'X', 'Y', 'Z']
chinese = ['zh_cuan', 'zh_e', 'zh_gan', 'zh_gan1', 'zh_gui', 'zh_gui1', 'zh_hei', 'zh_hu', 'zh_ji', 'zh_jin',
           'zh_jing', 'zh_jl', 'zh_liao', 'zh_lu', 'zh_meng', 'zh_min', 'zh_ning', 'zh_qing', 'zh_qiong',
           'zh_shan', 'zh_su', 'zh_sx', 'zh_wan', 'zh_xiang', 'zh_xin', 'zh_yu', 'zh_yu1', 'zh_yue', 'zh_yun',
           'zh_zang', 'zh_zhe']
dataset = numbers + alphbets + chinese


def list_all_files(root):
    files = []
    list = os.listdir(root)
    for i in range(len(list)):
        element = os.path.join(root, list[i])
        if os.path.isdir(element):
            temp_dir = os.path.split(element)[-1]
            if temp_dir in dataset:
                files.extend(list_all_files(element))
        elif os.path.isfile(element):
            files.append(element)
    return files


def init(dir):
    if not os.path.exists(dir):
        raise ValueError('没有找到文件夹')
    files = list_all_files(dir)
    for file in files:
        src_img = cv2.imread(file)
        af_img = Affine(src_img)
        cv2.imwrite(file.title() + "_af.jpg", af_img)


def Affine(img):
    rows, cols, channel = img.shape
    T = cv2.getRotationMatrix2D((cols / 2, rows / 2), 30, 0.8)
    dst = cv2.warpAffine(img, T, (cols, rows))
    return dst


if __name__ == '__main__':
    cur_dir = sys.path[0]
    data_dir = os.path.join(cur_dir, 'train/chars2')
    data_dir_chinese = os.path.join(cur_dir, 'train/charsChinese')
    test_dir = os.path.join(cur_dir, 'images/cnn_char_test')
    init(data_dir)
    init(data_dir_chinese)