# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2023/5/4  9:17
# @Author  : Cao Xu
# @FileName: data_preprocess.py
"""
Description:   
"""
import collections
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
    base_dir = r'D:\med dataset\kidney\zhongshan-kidney-json-20231204\malignant'
    a, b, c = 0, 0, 0
    for p in os.listdir(base_dir):
        jsons = [os.path.join(base_dir, p, x) for x in os.listdir(os.path.join(base_dir, p)) if x.endswith('.json')]
        for j in jsons:
            with open(j, 'r', encoding='utf-8') as fp:
                json_data = json.load(fp)
                try:
                    if len(json_data['shapes']) > 1:
                        for n in range(len(json_data['shapes'])):
                            print(j, '----', json_data['shapes'][n]['label'])
                        a += 1
                    elif len(json_data['shapes']) == 1 and json_data['shapes'][0]['label'] == 'Kidney' or \
                            json_data['shapes'][0]['label'] == 'renal' or json_data['shapes'][0]['label'] == 'Renal':
                        b += 1
                    elif len(json_data['shapes']) == 1 and json_data['shapes'][0]['label'] == 'Kidney cancer' \
                            or json_data['shapes'][0]['label'] == 'tumor' or json_data['shapes'][0]['label'] == 'Mass':
                        c += 1
                except:
                    print(j)
    print(a, b, c)


def kfold_split():
    base_dir = 'D:/MED_File/dataset/KidneyDataset/ori dataset/'
    jsons = [x for x in os.listdir(base_dir) if x.endswith('.json')]
    kidney_split = False
    cancer_split = True
    ben_list, mal_list = [], []
    if cancer_split:
        ori_out_path = 'D:/MED_File/dataset/KidneyDataset/kfold-cancer-ori/'
        if not os.path.exists(ori_out_path):
            os.makedirs(ori_out_path, exist_ok=True)
        crop_out_path = 'D:/MED_File/dataset/KidneyDataset/kfold-cancer-crop/'
        if not os.path.exists(crop_out_path):
            os.makedirs(crop_out_path, exist_ok=True)
        cancer_mask_out_path = 'D:/MED_File/dataset/KidneyDataset/kfold-cancer-mask/'
        if not os.path.exists(cancer_mask_out_path):
            os.makedirs(cancer_mask_out_path, exist_ok=True)
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
        ori_out_path = 'D:/MED_File/dataset/KidneyDataset/kfold-kidney-ori/'
        if not os.path.exists(ori_out_path):
            os.makedirs(ori_out_path, exist_ok=True)
        kidney_mask_out_path = 'D:/MED_File/dataset/KidneyDataset/kfold-kidney-mask/'
        if not os.path.exists(kidney_mask_out_path):
            os.makedirs(kidney_mask_out_path, exist_ok=True)
        for j in jsons:
            with open(base_dir + j, 'r', encoding='utf-8') as fp:
                json_data = json.load(fp)
                if len(json_data['shapes']) > 1:
                    mal_list.append(j)
                elif len(json_data['shapes']) == 1 and json_data['shapes'][0]['label'] == 'Kidney cancer':
                    ben_list.append(j)
                elif len(json_data['shapes']) == 1 and json_data['shapes'][0]['label'] == 'Kidney':
                    mal_list.append(j)

    for cla in ['benign', 'malignant']:
        if cla == 'benign':
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
            if cancer_split:
                crop_save_path = os.path.join(crop_out_path, f"fold{index}", cla)
                if not os.path.exists(crop_save_path):
                    os.makedirs(crop_save_path, exist_ok=True)
                cancer_mask_save_path = os.path.join(cancer_mask_out_path, f"fold{index}", cla)
                if not os.path.exists(cancer_mask_save_path):
                    os.makedirs(cancer_mask_save_path, exist_ok=True)
            if kidney_split:
                kidney_mask_save_path = os.path.join(kidney_mask_out_path, f"fold{index}", cla)
                if not os.path.exists(kidney_mask_save_path):
                    os.makedirs(kidney_mask_save_path, exist_ok=True)
            for img_name in img_list:
                if img_name in cross:
                    'copy ori image and json'
                    shutil.copy(os.path.join(base_dir, img_name), os.path.join(ori_save_path, img_name))  # crop json
                    shutil.copy(os.path.join(base_dir, img_name.replace('.json', '.jpg')),
                                os.path.join(ori_save_path, img_name.replace('.json', '.jpg')))  # crop image
                    img = cv2.imread(os.path.join(base_dir, img_name.replace('.json', '.jpg')))
                    with open(base_dir + img_name, 'r', encoding='utf-8') as fp:
                        json_data = json.load(fp)
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
                        'crop image'
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


def img2video():
    fps = 2
    img_dir = r'D:\med_project\中山医院-肾脏\20231228-dataset-mass-segment\good'
    img_list = [x for x in os.listdir(img_dir) if x.lower().endswith('.jpg')]
    # img_key = lambda i: int(i.split('.')[-1])  # .split('frame')[1]
    # img_list = sorted(os.listdir(img_dir), key=img_key)
    img1 = cv_read(os.path.join(img_dir, img_list[0]))
    img_size = (img1.shape[1], img1.shape[0])
    video_dir = r'D:\med_project\中山医院-肾脏\20231228-dataset-mass-segment'
    os.makedirs(video_dir, exist_ok=True)
    # MJPG --> .avi   mp4v -->.mp4
    video = cv2.VideoWriter(os.path.join(video_dir, 'renal-mass-segment.avi'),
                            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, img_size)
    for i in range(0, len(img_list) - 1):
        img = cv_read(os.path.join(img_dir, img_list[i]))
        if (img.shape[1], img.shape[0]) != img_size:
            img = cv2.resize(img, img_size)
        video.write(img)
    video.release()


def mead_split_patient():
    """
        按病例数分成5折
    """
    random.seed(0)

    # 设置５折实验

    org_path = 'D:/med_project/上海十院肾囊肿疾病/肾囊性病变/'
    out_path = 'D:/med_dataset/kidney/240122-renal-cystic-classify-5fold/'

    for cla in ['恶性', '良性']:
        in_path = os.path.join(org_path, cla)
        img_list = []
        patient_list = []
        for name in os.listdir(in_path):
            img_list.extend(os.listdir(os.path.join(org_path, cla, name)))
            if name not in patient_list:
                patient_list.append(name)

        random.shuffle(patient_list)  # 打乱病例名称
        img_nums = len(img_list)  # 所有的图片数目
        patient_nums = len(patient_list)  # 病例数
        temp = func(patient_list, int(patient_nums * 0.2), m=5)  # 平均分为5份,5折交叉训练

        for index, cross in enumerate(temp):
            print(" %d / %d " % (index + 1, img_nums))  # processing bar
            new_save_path = os.path.join(out_path, f"fold{index}", cla)
            os.makedirs(new_save_path, exist_ok=True)
            for patient in patient_list:
                if patient in cross:
                    shutil.copytree(os.path.join(org_path, cla, patient), os.path.join(new_save_path, patient))


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


def cv_write(file_path, file):
    cv2.imencode('.bmp', file)[1].tofile(file_path)


def get_mask_by_json():
    base_dir = 'D:/med_dataset/kidney/240122-renal-cystic-classify-5fold/'
    segment_mask = 'D:/med_dataset/kidney/240122-renal-cystic-segment-5fold/'
    for f in os.listdir(base_dir):
        for c in os.listdir(os.path.join(base_dir, f)):
            for p in os.listdir(os.path.join(base_dir, f, c)):
                os.makedirs(os.path.join(segment_mask, f, c, p), exist_ok=True)
                img_json_list = [x for x in os.listdir(os.path.join(base_dir, f, c, p)) if x.endswith('.json')]
                for img_json in img_json_list:
                    img = cv_read(os.path.join(base_dir, f, c, p, img_json.replace('.json', '.jpg')))
                    'mass'
                    seg_img = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

                    with open(os.path.join(base_dir, f, c, p, img_json), 'r', encoding='utf-8') as fp:
                        json_data = json.load(fp)
                    for i in range(len(json_data['shapes'])):
                        if json_data['shapes'][i]['label'] == '肾脏':
                            points = np.array(json_data['shapes'][i]['points'], np.int32)
                            cv2.fillConvexPoly(seg_img, points, 128)
                    for i in range(len(json_data['shapes'])):
                        if json_data['shapes'][i]['label'] in ['良性', '恶性']:
                            points = np.array(json_data['shapes'][i]['points'], np.int32)
                            cv2.fillConvexPoly(seg_img, points, 255)
                    'mass'
                    cv_write(os.path.join(segment_mask, f, c, p, img_json.replace('.json', '.jpg')), seg_img)


def dataset_augment():
    org_path = 'D:/med dataset/kidney/zhongshan-kidney-dataset/0/'
    for p in os.listdir(org_path):
        for i in os.listdir(os.path.join(org_path, p)):
            original_image = cv_read(os.path.join(org_path, p, i))
            rows, cols, _ = original_image.shape
            flip_flag = random.randint(-1, 1)
            bright_flag = random.uniform(1, 1.5)
            flipped_image = cv2.flip(original_image, flip_flag)  # 1表示水平翻转
            # 调整亮度，alpha为缩放因子，beta为偏移值
            brightened_image = cv2.convertScaleAbs(flipped_image, alpha=bright_flag, beta=0)
            # 平移
            x_translation = random.randint(-20, 20)
            y_translation = random.randint(-20, 20)
            translation_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
            translated_image = cv2.warpAffine(brightened_image, translation_matrix, (cols, rows))
            # 保存增强后的图像
            new_name = os.path.splitext(i)[0] + '-augment' + os.path.splitext(i)[1]
            cv_write(os.path.join(org_path, p, new_name), translated_image)


def image_json_compare():
    base_dir = r'D:\med_dataset\kidney\20231228-zhongshan-classify-json-5fold\fold3\1'
    for p in os.listdir(base_dir):
        json_list = [x for x in os.listdir(os.path.join(base_dir, p)) if x.endswith('.json')]
        for img_json in json_list:
            if os.path.exists(os.path.join(base_dir, p, img_json.replace('.json', '.jpg'))):
                img = cv_read(os.path.join(base_dir, p, img_json.replace('.json', '.jpg')))
                img_name = os.path.join(base_dir, p, img_json.replace('.json', '.jpg'))
            else:
                img = cv_read(os.path.join(base_dir, p, img_json.replace('.json', '.JPG')))
                img_name = os.path.join(base_dir, p, img_json.replace('.json', '.jpg'))

            with open(os.path.join(base_dir, p, img_json), 'r', encoding='utf-8') as fp:
                json_data = json.load(fp)
                json_height = json_data['imageHeight']
                json_width = json_data['imageWidth']
                image_height = img.shape[0]
                image_width = img.shape[1]
                if json_height != image_height and json_width != image_width:
                    print(os.path.join(base_dir, p, img_json))
                    img = cv2.resize(img, (json_height, json_width))
                    cv_write(img_name, img)
                    # print(os.path.join(base_dir, p, img_json))
                    # json_data['imageHeight'] = image_height
                    # json_data['imageWidth'] = image_width
                    # with open(os.path.join(base_dir, p, img_json), 'w', encoding='utf-8') as f:
                    #     json.dump(json_data, f)


if __name__ == '__main__':
    # dataset_count()
    # kfold_split()
    # img2video()
    # mead_split_patient()
    get_mask_by_json()
    # dataset_augment()
    # image_json_compare()

    # mask_img = cv2.imread('D:/med dataset/kidney-small-tumor-mask/fold1/21-result/Image01.JPG')
    # mask_img[mask_img == 64] = 1
    # mask_img[mask_img == 128] = 1
    # mask_img[mask_img == 255] = 1
    # renal_img, reference_img, mass_img = cv2.split(mask_img)
    # renal_img[renal_img == 1] = 64
    # reference_img[reference_img == 1] = 128
    # mass_img[mass_img == 1] = 255
    # imgMerge = cv2.merge([renal_img, reference_img, mass_img])
    # cv2.imwrite('test.png', imgMerge)

    # base_dir = r'D:\med_project\上海十院肾囊肿疾病\肾囊性病变'
    # for c in os.listdir(base_dir):
    #     for p in os.listdir(os.path.join(base_dir, c)):
    #         files = os.listdir(os.path.join(base_dir, c, p))
    #         # if (len(files) % 2) != 0:
    #         #     print(c, p)
    #         json_list = [x for x in os.listdir(os.path.join(base_dir, c, p)) if x.endswith('.json')]
    #         img_list = [x for x in os.listdir(os.path.join(base_dir, c, p)) if not x.endswith('.json')]
    #         if len(img_list) + len(json_list) != len(files):
    #             print(c, p)
    #         if len(img_list) != len(json_list):
    #             print(c, p)
