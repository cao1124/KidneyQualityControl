#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : RenalCysticDisease.py
@Author  : cao xu
@Time    : 2024/1/22 14:54
"""
import os
from enum import Enum

import torch
from classification import train as classify_train
from KidneySmallTumor import train as segment_train


def classify():
    os.environ['CUDA_VISIBLE_DEVICES'] = "4"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'resnext50'
    data_dir = '/media/user/Disk1/caoxu/dataset/kidney/240122-renal-cystic-classify-5fold/'
    category_num = 2
    bs = 128
    lr = 0.01
    num_epochs = 500
    data = 'RenalCysticDisease/240122-renal-cystic-classify-'
    save_path = data + str(category_num) + 'class-' + model_name + '-bs' + str(bs) + '-lr' + str(lr) + '/'
    pt_dir = 'RenalCysticDiseaseModel/classify/' + save_path
    if not os.path.exists(pt_dir):
        os.makedirs(pt_dir)

    print('测试肾囊肿{}分类,{}模型, batch size等于{}下的分类结果：'.format(category_num, model_name, bs))
    classify_train(data_dir, num_epochs, bs, pt_dir, category_num, model_name, device, lr)
    print('done')


def segment():
    os.environ['CUDA_VISIBLE_DEVICES'] = "6"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = '/media/user/Disk1/caoxu/dataset/kidney/240122-renal-cystic-classify-5fold/'
    encoder_name = "efficientnet-b7"
    encoder_activation = "softmax2d"
    target_list = [x.name for x in RenalCystic]
    bs = 6
    lr = 1e-4
    epochs = 5000
    save_dir = "RenalCysticDiseaseModel/segment/240122-renal-cystic-segment-" + encoder_name + '/'
    segment_train(data_dir, encoder_name, encoder_activation, bs, lr, epochs, save_dir, device, target_list)


class RenalCystic(Enum):
    renal = 1
    cystic = 2


if __name__ == '__main__':
    # classify()
    segment()

