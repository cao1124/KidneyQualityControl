#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：kidney-quality-control 
@File    ：data_prepare.py
@IDE     ：PyCharm 
@Author  ：cao xu
@Date    ：2025/6/3 上午10:21 
"""

import os
from enum import Enum

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


class TypeClass7(Enum):
    灰阶 = 0
    血流 = 1
    弹性 = 2
    造影 = 3
    频谱 = 4
    组织频谱 = 5
    心脏M型 = 6


def cv_read(file_path, flag=-1):
    # 可读取图片（路径为中文）
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flags=flag)
    # flag = -1,   8位深度，原通道
    # flag = 0，   8位深度，1通道
    # flag = 1，   8位深度，3通道
    # flag = 2，   原深度， 1通道
    # flag = 3，   原深度， 3通道
    # flag = 4，   8位深度，3通道
    return cv_img


def img_classify_cpu(img, model):
    with torch.no_grad():
        output = model(img)
        pred = torch.softmax(output, dim=1)
        cls = torch.argmax(pred).numpy()
    return cls, pred[0][cls].numpy()


def folder_rename():
    base_dir = r'F:\med_dataset\kidney_dataset\kidney-中山\20250530-病例汇总'
    for p in os.listdir(base_dir):
        if '-' in p:
            old_folder = os.path.join(base_dir, p)
            new_name = p.split('-')[0]
            new_folder = os.path.join(base_dir, new_name)
            os.rename(old_folder, new_folder)


def folder_match():
    excel_path = r'F:\med_dataset\kidney_dataset\kidney-中山\20250530-病例汇总.xlsx'
    data_df = pd.read_excel(excel_path)
    p_list = data_df['编号'].tolist()

    excel_path2 = r'F:\med_dataset\kidney_dataset\kidney-中山\20240419-复旦中山医院肾肿瘤病理编号1-600共508例.xlsx'
    data_df2 = pd.read_excel(excel_path2)
    num_list = data_df2['序号'].tolist()
    cls_list = data_df2['病理'].tolist()
    sex_list = data_df2['性别'].tolist()
    age_list = data_df2['年龄'].tolist()

    base_dir = r'F:\med_dataset\kidney_dataset\kidney-中山\20250530-病例汇总'
    # 准备要添加的新数据
    new_data = []
    columns = data_df.columns.tolist()
    out_excel = r'F:\med_dataset\kidney_dataset\kidney-中山\20250603-补充excel.xlsx'
    for p in os.listdir(base_dir):
        case_id = int(p)
        if case_id not in p_list:
            if case_id in num_list:
                idx = num_list.index(case_id)
                cls = cls_list[idx]
                sex = sex_list[idx]
                age = age_list[idx]
                new_row = [case_id, 'N', cls, sex, age, 'N', 'N', 'N', 'N']
                new_data.append(new_row)
            else:
                print('缺失： ', case_id)
    new_df = pd.DataFrame(new_data, columns=columns)
    new_df.to_excel(out_excel, index=False, engine='openpyxl')


def folder_type_rename():
    type_model = torch.load('D:/med_code/us-qc/模型整理/美年规则241202/20241204-type-qc-0.9822.pt', map_location='cpu')
    type_model.eval()
    base_dir = r'F:\med_dataset\kidney_dataset\kidney-中山\20250530-病例汇总2'
    for root, dirs, files in os.walk(base_dir):
        print(root)
        if len(files) > 2:
            print(root, '数量大于2')
            continue
        for name in files:
            img_ori = cv_read(os.path.join(root, name), flag=3)
            img = Image.fromarray(img_ori).convert('RGB')
            img = trans(img)
            img = torch.unsqueeze(img, dim=0)
            type_res, _ = img_classify_cpu(img, type_model)
            type_cls = TypeClass7(type_res).name
            if type_cls in ['灰阶', '血流']:
                p_num = root.split('\\')[-1]
                new_name = f'{p_num}-{type_cls}.jpg'
                if not os.path.exists(os.path.join(root, new_name)):
                    os.rename(os.path.join(root, name), os.path.join(root, new_name))
                else:
                    if type_cls == '血流':
                        os.rename(os.path.join(root, name), os.path.join(root, f'{p_num}-灰阶.jpg'))
                    else:
                        os.rename(os.path.join(root, name), os.path.join(root, f'{p_num}-血流.jpg'))
            else:
                print('模态问题： ', os.path.join(root, name))


if __name__ == '__main__':
    device = "cpu"
    mean, std = [0.29003, 0.29385, 0.31377], [0.18866, 0.19251, 0.19958]
    trans = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(), transforms.Normalize(mean, std)])
    # folder_rename()
    # folder_match()
    folder_type_rename()
    print('done.')
