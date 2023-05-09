# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2023/5/4  9:17
# @Author  : Cao Xu
# @FileName: data_preprocess.py
"""
Description:   
"""
import json
import os
import random
import shutil
import cv2
import numpy as np


def func(list_temp, n, m=5):
    """ listTemp 为列表 平分后每份列表的的个数"""
    count = 0
    for i in range(0, len(list_temp), n):
        count += 1
        if count == m:
            yield list_temp[i:]
            break
        else:
            yield list_temp[i:i + n]


def dataset_count():
    base_dir = 'D:/MED_File/dataset/KidneyDataset/ori dataset/'
    jsons = [x for x in os.listdir(base_dir) if x.endswith('.json')]
    a, b, c = 0, 0, 0
    for j in jsons:
        with open(base_dir + j, 'r', encoding='utf-8') as fp:
            json_data = json.load(fp)
            if len(json_data['shapes']) > 1:
                a += 1
            elif len(json_data['shapes']) == 1 and json_data['shapes'][0]['label'] == 'Kidney':
                b += 1
            elif len(json_data['shapes']) == 1 and json_data['shapes'][0]['label'] == 'Kidney cancer':
                c += 1
    print(a, b, c)


def kfold_split():
    base_dir = 'D:/MED_File/dataset/KidneyDataset/ori dataset/'
    jsons = [x for x in os.listdir(base_dir) if x.endswith('.json')]
    ori_out_path = 'D:/MED_File/dataset/KidneyDataset/kfold-ori/'
    if not os.path.exists(ori_out_path):
        os.makedirs(ori_out_path, exist_ok=True)
    crop_out_path = 'D:/MED_File/dataset/KidneyDataset/kfold-crop/'
    if not os.path.exists(crop_out_path):
        os.makedirs(crop_out_path, exist_ok=True)
    kidney_mask_out_path = 'D:/MED_File/dataset/KidneyDataset/kfold-kidney-mask/'
    if not os.path.exists(kidney_mask_out_path):
        os.makedirs(kidney_mask_out_path, exist_ok=True)
    cancer_mask_out_path = 'D:/MED_File/dataset/KidneyDataset/kfold-cancer-mask/'
    if not os.path.exists(cancer_mask_out_path):
        os.makedirs(cancer_mask_out_path, exist_ok=True)
    kidney_split = True
    cancer_split = False
    ben_list, mal_list = [], []
    if cancer_split:
        for j in jsons:
            with open(base_dir + j, 'r', encoding='utf-8') as fp:
                json_data = json.load(fp)
                if len(json_data['shapes']) > 1:
                    mal_list.append(j)
                elif len(json_data['shapes']) == 1 and json_data['shapes'][0]['label'] == 'Kidney':
                    ben_list.append(j)
                elif len(json_data['shapes']) == 1 and json_data['shapes'][0]['label'] == 'Kidney cancer':
                    mal_list.append(j)
    if kidney_split:
        for j in jsons:
            with open(base_dir + j, 'r', encoding='utf-8') as fp:
                json_data = json.load(fp)
                if len(json_data['shapes']) > 1:
                    mal_list.append(j)
                elif len(json_data['shapes']) == 1 and json_data['shapes'][0]['label'] == 'Kidney cancer':
                    ben_list.append(j)
                elif len(json_data['shapes']) == 1 and json_data['shapes'][0]['label'] == 'Kidney':
                    mal_list.append(j)

    for cla in ['0', '1']:
        if cla == '0':
            img_list = ben_list
        else:
            img_list = mal_list

        random.shuffle(img_list)  # 打乱
        img_nums = len(img_list)  # 所有的图片数目

        temp = func(img_list, int(img_nums * 0.2), m=5)  # 平均分为5份,5折交叉训练

        for index, cross in enumerate(temp):
            print(" %d / %d " % (index + 1, img_nums))  # processing bar
            ori_save_path = os.path.join(ori_out_path, f"fold{index}", cla)
            if not os.path.exists(ori_save_path):
                os.makedirs(ori_save_path, exist_ok=True)
            crop_save_path = os.path.join(crop_out_path, f"fold{index}", cla)
            if not os.path.exists(crop_save_path):
                os.makedirs(crop_save_path, exist_ok=True)
            kidney_mask_save_path = os.path.join(kidney_mask_out_path, f"fold{index}", cla)
            if not os.path.exists(kidney_mask_save_path):
                os.makedirs(kidney_mask_save_path, exist_ok=True)
            cancer_mask_save_path = os.path.join(cancer_mask_out_path, f"fold{index}", cla)
            if not os.path.exists(cancer_mask_save_path):
                os.makedirs(cancer_mask_save_path, exist_ok=True)
            for img_name in img_list:
                if img_name in cross:
                    'copy ori image and json'
                    shutil.copy(os.path.join(base_dir, img_name), os.path.join(ori_save_path, img_name))
                    shutil.copy(os.path.join(base_dir, img_name.replace('.json', '.jpg')),
                                os.path.join(ori_save_path, img_name.replace('.json', '.jpg')))
                    'crop image'
                    img = cv2.imread(os.path.join(base_dir, img_name.replace('.json', '.jpg')))
                    with open(base_dir + img_name, 'r', encoding='utf-8') as fp:
                        json_data = json.load(fp)
                        if len(json_data['shapes']) > 1:
                            if json_data['shapes'][0]['label'] == 'Kidney cancer':
                                points = np.array(json_data['shapes'][0]['points'])
                            else:
                                points = np.array(json_data['shapes'][1]['points'])
                        else:
                            points = np.array(json_data['shapes'][0]['points'])
                    crop_img = img[int(min(points[:, 1])): int(max(points[:, 1])),
                                   int(min(points[:, 0])): int(max(points[:, 0]))]
                    cv2.imwrite(os.path.join(crop_save_path, img_name.replace('.json', '.png')), crop_img)
                    'kidney mask image'
                    if kidney_split:
                        kidney_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                        if len(json_data['shapes']) == 1 and json_data['shapes'][0]['label'] == 'Kidney':
                            points = np.array(json_data['shapes'][0]['points'])
                            polygon = np.array(points, np.int32)  # 坐标为顺时针方向
                            cv2.fillConvexPoly(kidney_mask, polygon, (255, 255, 255))
                        elif len(json_data['shapes']) > 1:
                            if json_data['shapes'][0]['label'] == 'Kidney':
                                points = np.array(json_data['shapes'][0]['points'])
                            else:
                                points = np.array(json_data['shapes'][1]['points'])
                            polygon = np.array(points, np.int32)  # 坐标为顺时针方向
                            cv2.fillConvexPoly(kidney_mask, polygon, (255, 255, 255))
                        cv2.imwrite(os.path.join(kidney_mask_save_path, img_name.replace('.json', '.png')), kidney_mask)
                    'cancer mask image'
                    if cancer_split:
                        cancer_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                        if len(json_data['shapes']) == 1 and json_data['shapes'][0]['label'] == 'Kidney cancer':
                            points = np.array(json_data['shapes'][0]['points'])
                            polygon = np.array(points, np.int32)  # 坐标为顺时针方向
                            cv2.fillConvexPoly(cancer_mask, polygon, (255, 255, 255))
                        elif len(json_data['shapes']) > 1:
                            if json_data['shapes'][0]['label'] == 'Kidney cancer':
                                points = np.array(json_data['shapes'][0]['points'])
                            else:
                                points = np.array(json_data['shapes'][1]['points'])
                            polygon = np.array(points, np.int32)  # 坐标为顺时针方向
                            cv2.fillConvexPoly(cancer_mask, polygon, (255, 255, 255))
                        cv2.imwrite(os.path.join(cancer_mask_save_path, img_name.replace('.json', '.png')), cancer_mask)


if __name__ == '__main__':
    # dataset_count()
    kfold_split()
