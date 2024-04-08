#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : RenalCysticDisease.py
@Author  : cao xu
@Time    : 2024/1/22 14:54
"""
import copy
import os
from enum import Enum

import cv2
import numpy as np
import torch
from classification import train as classify_train
from KidneySmallTumor import train as segment_train
from data_preprocess import cv_read, cv_write
from segment_util import add_weighted, get_iou, get_f1


def classify():
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'resnet50'
    data_dir = '/media/user/Disk1/caoxu/dataset/kidney/shiyuan/20240122-renal-cystic-crop-classify-5fold/'
    category_num = 2
    bs = 128
    lr = 0.01
    num_epochs = 500
    data = '20240408-renal-cystic-image-crop-classify-'
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
    epochs = 2000
    save_dir = "RenalCysticDiseaseModel/segment/240122-renal-cystic-segment-" + encoder_name + '/'
    segment_train(data_dir, encoder_name, encoder_activation, bs, lr, epochs, save_dir, device, target_list)


class RenalCystic(Enum):
    renal = 1
    cystic = 2


def segment_test():
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_list = []
    test_img_dir = r'D:\med_dataset\kidney\240122-renal-cystic-classify-5fold\fold2'
    for c in os.listdir(test_img_dir):
        for p in os.listdir(os.path.join(test_img_dir, c)):
            for img_path in [x for x in os.listdir(os.path.join(test_img_dir, c, p)) if not x.endswith('json')]:
                test_list.append(os.path.join(test_img_dir, c, p, img_path))
    save_dir = r'D:\med_project\上海十院肾囊肿疾病\fold1-model-pred'
    os.makedirs(save_dir, exist_ok=True)
    model = torch.load('D:/med_project/上海十院肾囊肿疾病/fold1-best_0.599.pth')
    model.eval()
    torch.cuda.empty_cache()
    target_list = [x.name for x in RenalCystic]
    color_list = ['BG', 'GR']
    iou_list, dice_list = [], []
    renal_iou_list, renal_dice_list = [], []
    mass_iou_list, mass_dice_list = [], []
    for k in range(len(test_list)):
        img_iou, img_dice = [], []
        image_ori = cv_read(test_list[k])
        gt_mask = cv_read(os.path.splitext(test_list[k].replace('classify', 'segment'))[0] + '.jpg')
        [orig_h, orig_w, _] = image_ori.shape
        image = cv2.resize(image_ori, (512, 512), interpolation=cv2.INTER_NEAREST)
        image = image / 255.0
        image = image.transpose(2, 0, 1).astype('float32')
        x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
        img_pred = copy.deepcopy(image_ori)
        img_gt = copy.deepcopy(image_ori)
        with torch.no_grad():
            pr_mask = model(x_tensor).squeeze(0).cpu().numpy()
        for c in range(1, len(target_list) + 1):
            mask_gt = copy.deepcopy(gt_mask)
            if c == 1:
                mask_gt[mask_gt != 128] = 0
                mask_gt[mask_gt == 128] = 255
            else:
                mask_gt[mask_gt != 255] = 0
                mask_gt[mask_gt == 255] = 255
            pred_mask = pr_mask[c]
            pred_mask[pred_mask < 0.5] = 0
            pred_mask[pred_mask >= 0.5] = 255
            if pred_mask.shape != image_ori.shape:
                pred_mask = cv2.resize(pred_mask, (orig_w, orig_h), cv2.INTER_NEAREST)
                _, pred_mask = cv2.threshold(pred_mask, 1, 255, cv2.THRESH_BINARY)
            img_pred = add_weighted(img_pred, pred_mask.astype('uint8'), color_list[c - 1])

            if np.sum(mask_gt) == 0 and np.sum(pred_mask) == 0:
                iou = 1
                dice = 1
            else:
                iou = np.round(get_iou(mask_gt, pred_mask), 4)
                dice = np.round(get_f1(mask_gt, pred_mask), 4)
            iou_list.append(iou)
            dice_list.append(dice)
            img_iou.append(iou)
            img_dice.append(dice)
            if c == 1:
                renal_iou_list.append(iou)
                renal_dice_list.append(dice)
            elif c == 2:
                mass_iou_list.append(iou)
                mass_dice_list.append(dice)
            img_gt = add_weighted(img_gt, mask_gt.astype('uint8'), color_list[c - 1])
        save_full_path = os.path.join(save_dir, str(np.round(np.average(img_iou), 4)) + "-" + test_list[k].split('\\')[-1])
        img_cat = np.concatenate((img_gt, img_pred), axis=1)
        cv_write(save_full_path, img_cat)

    print("\tRenal Mean Dice: ", np.round(np.average(renal_dice_list), 4),
          "Renal Mean IoU:", np.round(np.average(renal_iou_list), 4))
    print("\tMass Mean Dice: ", np.round(np.average(mass_dice_list), 4),
          "Mass Mean IoU:", np.round(np.average(mass_iou_list), 4))
    print("\tAll Mean Dice: ", np.round(np.average(dice_list), 4),
          "All Mean IoU:", np.round(np.average(iou_list), 4))


if __name__ == '__main__':
    classify()
    # segment()
    # segment_test()
