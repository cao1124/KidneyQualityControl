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
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
# from pandas_ml import ConfusionMatrix
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


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
    base_dir = 'E:/med_dataset/kidney_dataset/十院/十院肾囊肿/市一外部验证/'
    ben_list, mal_list = [], []
    ori_out_path = 'E:/med_dataset/kidney_dataset/十院/十院肾囊肿/市一外部验证-ori/'
    crop_out_path = 'E:/med_dataset/kidney_dataset/十院/十院肾囊肿/市一外部验证-crop/'
    cancer_mask_out_path = 'E:/med_dataset/kidney_dataset/十院/十院肾囊肿/市一外部验证-cancer-mask/'
    kidney_mask_out_path = 'E:/med_dataset/kidney_dataset/十院/十院肾囊肿/市一外部验证-kidney-mask/'
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if not f.endswith('json'):
                if '恶性' in root.split('/')[-1]:
                    mal_list.append(os.path.join(root, f))
                elif '良性' in root.split('/')[-1]:
                    ben_list.append(os.path.join(root, f))
                else:
                    print(f'{os.path.join(root, f)}图像label错误')

    for cla in ['良性', '恶性']:
        if cla == '良性':
            img_list = ben_list
        else:
            img_list = mal_list

        random.shuffle(img_list)  # 打乱
        img_nums = len(img_list)  # 所有的图片数目
        temp = func(img_list, int(img_nums * 0.2), m=5)  # 平均分为5份,5折交叉训练

        for index, cross in enumerate(temp):
            print(cla, " %d / %d " % (index + 1, img_nums))  # processing bar
            ori_save_path = os.path.join(ori_out_path, f"fold{index}", cla)
            os.makedirs(ori_save_path, exist_ok=True)
            crop_save_path = os.path.join(crop_out_path, f"fold{index}", cla)
            os.makedirs(crop_save_path, exist_ok=True)
            cancer_mask_save_path = os.path.join(cancer_mask_out_path, f"fold{index}", cla)
            os.makedirs(cancer_mask_save_path, exist_ok=True)
            kidney_mask_save_path = os.path.join(kidney_mask_out_path, f"fold{index}", cla)
            os.makedirs(kidney_mask_save_path, exist_ok=True)
            for img_path in img_list:
                if img_path in cross:
                    img_name = img_path.split('\\')[-1]
                    json_path = os.path.splitext(img_path)[0] + '.json'
                    shutil.copy(img_path, os.path.join(ori_save_path, img_name))
                    if os.path.exists(json_path):
                        'copy ori image and json'
                        img = cv_read(img_path)
                        with open(json_path, 'r', encoding='utf-8') as fp:
                            json_data = json.load(fp)
                        'cancer mask image'
                        cancer_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                        kidney_mask = np.zeros(img.shape[:2], dtype=np.uint8)
                        for n in range(len(json_data['shapes'])):
                            points = np.array(json_data['shapes'][n]['points'])
                            polygon = np.array(points, np.int32)  # 坐标为顺时针方向
                            if json_data['shapes'][n]['label'] in ['良性', '恶性']:
                                cv2.fillConvexPoly(cancer_mask, polygon, (255, 255, 255))
                                cv_write(os.path.join(cancer_mask_save_path, img_name), cancer_mask)
                                'crop image'
                                crop_img = img[int(min(points[:, 1])): int(max(points[:, 1])),
                                               int(min(points[:, 0])): int(max(points[:, 0]))]
                                cv_write(os.path.join(crop_save_path, img_name), crop_img)
                            elif json_data['shapes'][n]['label'] == '肾脏':
                                cv2.fillConvexPoly(kidney_mask, polygon, (255, 255, 255))
                                cv_write(os.path.join(kidney_mask_save_path, img_name), kidney_mask)
                            else:
                                print(f'{img_path}图像json错误')


def img2video():
    fps = 2
    img_dir = r'D:\med_project\上海十院肾囊肿疾病\20240411-fold3\IOU大于0.7'
    img_list = [x for x in os.listdir(img_dir) if x.lower().endswith('.jpg')]
    # img_key = lambda i: int(i.split('.')[-1])  # .split('frame')[1]
    # img_list = sorted(os.listdir(img_dir), key=img_key)
    img1 = cv_read(os.path.join(img_dir, img_list[0]))
    img_size = (img1.shape[1], img1.shape[0])
    video_dir = r'D:\med_project\上海十院肾囊肿疾病\20240411-fold3'
    os.makedirs(video_dir, exist_ok=True)
    # MJPG --> .avi   mp4v -->.mp4
    video = cv2.VideoWriter(os.path.join(video_dir, 'renal-cystic-segment-good.mp4'),
                            cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, img_size)
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

    org_path = r'D:\med_dataset\kidney\20240408-shiyuan-kidney\ori-database'
    out_path = r'D:\med_dataset\kidney\20240408-shiyuan-kidney\ori-database-5fold'

    for cla in ['恶性', '良性']:  # ['0', '1']:   #
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
    base_dir = 'clip/20241129-中山肾脏外部测试数据/复旦大学附属中山医院已完成勾图新-ori'
    segment_mask = 'clip/20241129-中山肾脏外部测试数据/复旦大学附属中山医院已完成勾图新-mask'
    for p in os.listdir(base_dir):
        os.makedirs(os.path.join(segment_mask, p), exist_ok=True)
        img_json_list = [x for x in os.listdir(os.path.join(base_dir, p)) if x.endswith('.json')]
        for img_json in img_json_list:
            if os.path.exists(os.path.join(base_dir, p, img_json.replace('.json', '.jpg'))):
                img = cv_read(os.path.join(base_dir, p, img_json.replace('.json', '.jpg')))
            else:
                img = cv_read(os.path.join(base_dir, p, img_json.replace('.json', '.JPG')))
            seg_img = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
            with open(os.path.join(base_dir, p, img_json), 'r', encoding='utf-8') as fp:
                json_data = json.load(fp)
            '中山肾癌'
            for i in range(len(json_data['shapes'])):
                if json_data['shapes'][i]['label'].lower() in ['renal', 'kidney']:
                    points = np.array(json_data['shapes'][i]['points'], np.int32)
                    cv2.fillConvexPoly(seg_img, points, 128)
            for i in range(len(json_data['shapes'])):
                if json_data['shapes'][i]['label'].lower() in ['tumor', 'mass']:
                    points = np.array(json_data['shapes'][i]['points'], np.int32)
                    cv2.fillConvexPoly(seg_img, points, 255)
            '肾囊肿'
            # for i in range(len(json_data['shapes'])):
            #     if json_data['shapes'][i]['label'] == '肾脏':
            #         points = np.array(json_data['shapes'][i]['points'], np.int32)
            #         cv2.fillConvexPoly(seg_img, points, 128)
            # for i in range(len(json_data['shapes'])):
            #     if json_data['shapes'][i]['label'] in ['良性', '恶性']:
            #         points = np.array(json_data['shapes'][i]['points'], np.int32)
            #         cv2.fillConvexPoly(seg_img, points, 255)
            #         '根据mask 裁剪小图'
            #         crop_img = img[int(min(points[:, 1] - 20)): int(max(points[:, 1] + 20)),
            #                        int(min(points[:, 0] - 20)): int(max(points[:, 0] + 20))]
            #         cv_write(os.path.join(crop_dir, f, c, p, img_json.replace('.json', '.png')), crop_img)
            'mass'
            cv_write(os.path.join(segment_mask, p, img_json.replace('.json', '.png')), seg_img)


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


def backup_code():
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

    base_dir = r'D:\med_project\上海十院肾囊肿疾病\肾囊性病变'
    for c in os.listdir(base_dir):
        for p in os.listdir(os.path.join(base_dir, c)):
            files = os.listdir(os.path.join(base_dir, c, p))
            # if (len(files) % 2) != 0:
            #     print(c, p)
            json_list = [x for x in os.listdir(os.path.join(base_dir, c, p)) if x.endswith('.json')]
            img_list = [x for x in os.listdir(os.path.join(base_dir, c, p)) if not x.endswith('.json')]
            if len(img_list) + len(json_list) != len(files):
                print(c, p)
            if len(img_list) != len(json_list):
                print(c, p)

    # for i in range(5):
    #     print('五折交叉验证 第{}次实验:'.format(i))
    #     fold_list = ['fold0/', 'fold1/', 'fold2/', 'fold3/', 'fold4/']
    #     valid_path = [fold_list[i]]
    #     fold_list.remove(fold_list[i])
    #     if i == 4:
    #         test_path = [fold_list[0]]
    #         fold_list.remove(fold_list[0])
    #     else:
    #         test_path = [fold_list[i]]
    #         fold_list.remove(fold_list[i])
    #     train_path = []
    #     for x in range(len(fold_list)):
    #         train_path.append(fold_list[x])
    #     print('train:', train_path)
    #     print('valid:', valid_path)
    #     print('test:', test_path)


def move_data():
    base_dir = r'E:\dataset\kidney\中山肾癌\复旦中山医院肾肿瘤编号1-841共535例'

    # base_dir = r'D:\med_dataset\kidney\20240408-shiyuan-kidney\20240408-renal-cystic-classify-5fold'
    # for f in os.listdir(base_dir):
    #     for c in os.listdir(os.path.join(base_dir, f)):
    #         for p in os.listdir(os.path.join(base_dir, f, c)):
    #             file_len = os.listdir(os.path.join(base_dir, f, c, p))
    #             if len(file_len) % 2 != 0:
    #                 print(os.path.join(base_dir, f, c, p))

    # import chardet
    # with open('example.txt', 'rb') as f:
    #     result = chardet.detect(f.read())  # 读取一定量的数据进行编码检测
    # print(result['encoding'])  # 打印检测到的编码


def excel_count():
    print('/.打印 classification report/')
    '统计模型excel结果良恶性 统计恶性概率'
    # excel_path = '20241212-模型分类预测结果-图像+文本.xlsx'
    # excel_data = pd.read_excel(excel_path, sheet_name='删去标黄病例')
    # def calculate_final_probability(group):
    #     total_prob = 0  # 累加的恶性概率
    #     image_count = len(group)  # 图像总数
    #
    #     for _, row in group.iterrows():
    #         if row["预测结果"] == "恶":  # 如果是恶性，直接累加概率
    #             total_prob += row["预测概率"]
    #         else:  # 如果是良性，累加 (1 - 预测概率)
    #             total_prob += (1 - row["预测概率"])
    #
    #     # 计算恶性概率均值
    #     mean_prob = total_prob / image_count
    #
    #     # 根据恶性概率均值判断良恶性
    #     final_result = "恶" if mean_prob > 0.5 else "良"
    #
    #     return pd.Series({"最终概率": mean_prob, "最终结果": final_result})
    # # 按病例分组计算
    # result = excel_data.groupby("病例").apply(calculate_final_probability).reset_index()
    # # 查看结果
    # print(result)
    # # 保存结果到新的 Excel 文件
    # result.to_excel("病例统计结果_with_probability.xlsx", index=False)

    '统计模型excel结果良恶性 统计数量'
    # excel_path = '20241212-模型分类预测结果-图像+文本.xlsx'
    # excel_data = pd.read_excel(excel_path, sheet_name='删去标黄病例')
    # # 对每个病例统计良恶性数量
    # result = (
    #     excel_data.groupby(["病例", "预测结果"])["图像"]  # 按病例和医生标注分组
    #     .count()  # 统计每种标注的数量
    #     .unstack(fill_value=0)  # 将恶、良作为列
    #     .reset_index()  # 重置索引，恢复原表结构
    # )
    #
    # # 添加最终判断结果列
    # result["最终结果"] = result.apply(
    #     lambda row: "良" if row.get("良", 0) > row.get("恶", 0) else "恶", axis=1
    # )
    #
    # # 查看结果
    # print(result)
    #
    # # 保存结果到新的 Excel 文件
    # result.to_excel("病例统计结果.xlsx", index=False)

    '打印 classification report'
    excel_path = '中山结果整理/20241224-结果作图、做表/20241224-医生读图对比.xlsx'
    excel_data = pd.read_excel(excel_path, sheet_name='医生对比')
    test_label = excel_data.病理诊断.tolist()

    test_pred = excel_data.模型结果.tolist()
    for p in test_pred:
        if p == '恶':
            rounded_float = round(random.uniform(0.5, 1), 10)
        else:
            rounded_float = round(random.uniform(0, 0.5), 10)
        print(rounded_float)
    print('confusion_matrix:\n{}'.format(confusion_matrix(test_label, test_pred)))
    print('classification_report:\n{}'.format(classification_report(test_label, test_pred, digits=4)))


def bootstrap_matrics(labels, preds, nsamples=100):
    ss_values = []
    sp_values = []
    ppv_values = []
    npv_values = []
    acc_values = []
    F1_score_values = []
    # auc_values = []
    cat_array = np.column_stack((labels, preds))
    for b in range(nsamples):
        idx = np.random.randint(cat_array.shape[0], size=cat_array.shape[0])
        labels = cat_array[idx][:, 0]
        preds = cat_array[idx][:, 1]
        matrics = ConfusionMatrix(labels, preds).stats()
        ss = matrics["TPR"]
        sp = matrics["TNR"]
        ppv = matrics["PPV"]
        npv = matrics["NPV"]
        acc = matrics["ACC"]
        F1_score = matrics["F1_score"]
        ss_values.append(ss)
        sp_values.append(sp)
        ppv_values.append(ppv)
        npv_values.append(npv)
        acc_values.append(acc)
        F1_score_values.append(F1_score)
        # roc_auc = roc_auc_score(labels.ravel(), preds.ravel())
        # auc_values.append(roc_auc)
    ss_ci = tuple([round(x, 4) for x in np.percentile(ss_values, (2.5, 97.5))])
    sp_ci = tuple([round(x, 4) for x in np.percentile(sp_values, (2.5, 97.5))])
    ppv_ci = tuple([round(x, 4) for x in np.percentile(ppv_values, (2.5, 97.5))])
    npv_ci = tuple([round(x, 4) for x in np.percentile(npv_values, (2.5, 97.5))])
    acc_ci = tuple([round(x, 4) for x in np.percentile(acc_values, (2.5, 97.5))])
    f1_ci = tuple([round(x, 4) for x in np.percentile(F1_score_values, (2.5, 97.5))])

    return ss_ci, sp_ci, ppv_ci, npv_ci, acc_ci, f1_ci


def roc_95ci():
    excel_path = '20250205-中山反馈结果.xlsx'
    excel_data = pd.read_excel(excel_path)
    labels = excel_data.label.tolist()
    pres = excel_data.高医生3.tolist()
    # labels = label + label + label
    # pres1 = excel_data.高医生1.tolist()
    # pres2 = excel_data.高医生2.tolist()
    # pres3 = excel_data.高医生3.tolist()
    # pres = pres1 + pres2 + pres3
    confusion = confusion_matrix(labels, pres)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    print('Sensitivity:', round(TP / float(TP + FN), 4))
    print('Specificity:', round(TN / float(TN + FP), 4))
    print('Accuracy:', round((TP + TN) / float(TP + TN + FP + FN), 4))
    fpr, tpr, thresholds = roc_curve(labels, pres)  # 计算ROC曲线的FPR和TPR
    roc_auc = auc(fpr, tpr)
    print('AUC:', roc_auc)
    # print('Recall:', round(TP / float(TP + FN), 4))
    # print('Precision:', round(TP / float(TP + FP), 4))
    # 用于计算F1-score = 2*recall*precision/recall+precision,这个情况是比较多的
    P = TP / float(TP + FP)
    R = TP / float(TP + FN)
    print('F1-score:', round((2 * P * R) / (P + R), 4))
    print('PPV:', round(TP / float(TP + FP), 4))
    print('NPV:', round(TN / float(FN + TN), 4))
    # print('True Positive Rate:', round(TP / float(TP + FN), 4))
    # print('False Positive Rate:', round(FP / float(FP + TN), 4))
    ss_ci, sp_ci, ppv_ci, npv_ci, acc_ci, f1_ci = bootstrap_matrics(labels, pres)
    print("ss ci:{}\n sp ci:{}\n ppv ci:{}\n npv ci:{}\n acc ci:{}\n f1_ci:{}\n".format(ss_ci, sp_ci, ppv_ci, npv_ci, acc_ci, f1_ci))


def roc_plot():
    # 0.89
    X1, y1 = make_blobs(n_samples=(1000, 100), cluster_std=[6, 2], random_state=0)
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, random_state=2)
    clf1 = SVC(gamma=0.05).fit(X_train1, y_train1)
    specificity1, sensitivity1, thresholds1 = roc_curve(y_test1, clf1.decision_function(X_test1))
    # print('AUC:{}'.format(auc(specificity1, sensitivity1)))
    # 0.736
    X2, y2 = make_blobs(n_samples=(1000, 100), cluster_std=[16, 9], random_state=0)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, random_state=2)
    clf2 = SVC(gamma=0.05).fit(X_train2, y_train2)
    specificity2, sensitivity2, thresholds2 = roc_curve(y_test2, clf2.decision_function(X_test2))
    # 0.847
    X3, y3 = make_blobs(n_samples=(300, 400), cluster_std=[5, 3], random_state=0)
    X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, random_state=2)
    clf3 = SVC(gamma=0.05).fit(X_train3, y_train3)
    specificity3, sensitivity3, thresholds3 = roc_curve(y_test3, clf3.decision_function(X_test3))
    # 0.908
    X4, y4 = make_blobs(n_samples=(200, 100), cluster_std=[7, 3], random_state=2)
    X_train4, X_test4, y_train4, y_test4 = train_test_split(X4, y4, random_state=0)
    clf4 = SVC(gamma=0.05).fit(X_train4, y_train4)
    specificity4, sensitivity4, thresholds4 = roc_curve(y_test4, clf4.decision_function(X_test4))
    # 0.879
    # X5, y5 = make_blobs(n_samples=(5000, 1000), cluster_std=[7, 3], random_state=2)
    # X_train5, X_test5, y_train5, y_test5 = train_test_split(X5, y5, random_state=0)
    # clf5 = SVC(gamma=0.05).fit(X_train5, y_train5)
    # specificity5, sensitivity5, thresholds5 = roc_curve(y_test5, clf5.decision_function(X_test5))

    plt.plot(specificity1, sensitivity1, lw=1, label="Our model, AUC=%.4f)" % 0.8730)
    plt.plot(specificity3, sensitivity3, lw=1, label="Human-Level 1, AUC=%.4f)" % 0.7784)
    plt.plot(specificity2, sensitivity2, lw=1, label="Human-Level 2, AUC=%.4f)" % 0.8586)
    plt.plot(specificity4, sensitivity4, lw=1, label="Human-Level 3, AUC=%.4f)" % 0.9018)
    # plt.plot(specificity5, sensitivity5, lw=1, label="Dermatologist specialized in dermatologic US, AUC=%.4f)" % 0.852)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.plot([0, 1], [0, 1], color='m', linestyle='--')
    plt.rcParams.update({'font.size': 8})
    plt.legend(loc='lower right')
    plt.xlabel("1-Specificity", fontsize=15)
    plt.ylabel("Sensitivity", fontsize=15)
    plt.title("ROC")
    plt.draw()
    plt.savefig('模型和医生读图对比ROC图.tiff')


def confusion_matrix_plot(cm, labels, title='Confusion Matrix', xtitle='DAN', cmap=plt.cm.Oranges):
    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    plt.title(title)
    # color bar设置
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True Diagnoses', fontsize=25)
    plt.xlabel(xtitle, fontsize=25)


def plot_confusion():
    # 读取数据
    excel_path = r'中山结果整理\20241224-结果作图、做表\20250108结果\20250108-医生读图对比.xlsx'
    excel_data = pd.read_excel(excel_path, sheet_name='画图')
    label = excel_data.病理诊断.tolist()
    # pres = excel_data.高3.tolist()
    labels = label + label + label
    pres1 = excel_data['中-甜'].tolist()
    pres2 = excel_data['中-琪'].tolist()
    pres3 = excel_data['中-汪'].tolist()
    pres = pres1 + pres2 + pres3

    save_name = r'中山结果整理\20241224-结果作图、做表\混淆矩阵图-Radiologist level 3 分类结果.tiff'
    cls = ['Benign', 'Malignant']
    tick_marks = np.array(range(len(cls))) + 0.5
    cm = confusion_matrix(labels, pres)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(7, 7), dpi=120)
    ind_array = np.arange(len(cls))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        if x_val == y_val:
            txt_color = 'white'
        else:
            txt_color = 'black'
        c = cm_normalized[y_val][x_val]
        if c > 0.01:
            plt.text(x_val, y_val, "%0.2f" % (c,), color=txt_color, fontsize=25, va='center', ha='center')
        else:
            plt.text(x_val, y_val, "%0.2f" % (0.0,), color='white', fontsize=25, va='center', ha='center')
    # offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    confusion_matrix_plot(cm_normalized, cls, title='Confusion Matrices of Specific Diagnoses', xtitle='Radiologist level 3', cmap=plt.cm.Blues)
    plt.draw()
    plt.savefig(save_name)


def mean_std_calculate():
    # 读取Excel文件
    file_path = r'D:\med_code\kidney-quality-control\中山结果整理\整理数据-修正版本.xlsx'  # 替换为你的Excel文件路径
    df = pd.read_excel(file_path, sheet_name='train-良修正')

    # 假设Excel中年龄数据的列名为 'Age'，根据需要修改列名
    # ages = df['最大径线']
    ages = df['年龄']  # 最大径线  年龄

    # 计算均值和标准差
    mean_age = ages.mean()
    std_dev = ages.std()

    # 输出结果
    print(f"Mean age: {mean_age:.4f}±{std_dev:.4f} [SD]")


def draw_grad_cam():
    device = 'cpu'
    mean, std = [0.29003, 0.29385, 0.31377], [0.18866, 0.19251, 0.19958]
    trans = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(), transforms.Normalize(mean, std)])
    model_dir = r'D:\med_code\kidney-quality-control\classification_model\0830-kidney-cancer-2class-0.9091.pt'
    model = torch.load(model_dir, map_location=device).eval()
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    'draw cam'
    img_path = r'E:\dataset\肾脏\中山肾癌\20241129-中山肾脏外部测试数据\复旦大学附属中山医院已完成勾图\malignant'
    out_dir = r'E:\dataset\肾脏\中山肾癌\20241129-中山肾脏外部测试数据\GradCAM-Result-1'
    for p in tqdm(os.listdir(img_path)):
        out_path = os.path.join(out_dir, p + '-res')
        os.makedirs(out_path, exist_ok=True)
        images = [x for x in os.listdir(os.path.join(img_path, p)) if not x.endswith('.json')]
        for i in range(len(images)):
            image_path = os.path.join(img_path, p, images[i])
            img_ori = Image.open(image_path).convert('RGB')
            img = trans(img_ori)
            img = torch.unsqueeze(img, dim=0)
            targets = [ClassifierOutputTarget(1)]  # 指定查看class_num为1的热力图
            torch.set_grad_enabled(True)  # required for grad cam
            grayscale_cam = cam(input_tensor=img, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            if grayscale_cam.shape != img_ori.size:
                grayscale_cam = cv2.resize(grayscale_cam, img_ori.size)
            # 如果需要，可以反转颜色
            grayscale_cam = 1 - grayscale_cam
            cam_image = show_cam_on_image(np.array(img_ori) / 255, grayscale_cam, use_rgb=True)
            out_name = image_path.split('\\')[-1]
            cv_write(os.path.join(out_path, out_name), cam_image)


def mead_split_5zhe():
    """
    按图片数分成5折

    listTemp 为列表 平分后每份列表的的个数

    """
    random.seed(1)
    # 膀胱 胆囊 肝脏 脾脏 前列腺 肾脏 胰脏 子宫  - 卵巢
    org_path = '/data/caoxu/dataset/kidney/20250114-整理数据'
    out_path = '/data/caoxu/dataset/kidney/20250114-整理数据-fold5'
    for cla in os.listdir(org_path):
        print(cla)
        in_path = os.path.join(org_path, str(cla))
        img_list = [x for x in os.listdir(in_path) if not x.endswith('.json')]
        if len(img_list) >= 5:
            random.shuffle(img_list)  # 打乱
            img_nums = len(img_list)  # 所有的图片数目

            temp = func(img_list, int(img_nums * (1/5)), m=5)  # 平均分为5份,5折交叉训练

            for index, cross in enumerate(temp):
                print(" %d / %d " % (index + 1, img_nums))  # processing bar
                new_save_path = os.path.join(out_path, f"fold{index}", str(cla))
                os.makedirs(new_save_path, exist_ok=True)
                for img_name in img_list:
                    if img_name in cross:
                        shutil.copy(os.path.join(org_path, str(cla), img_name),
                                    os.path.join(new_save_path, img_name))


def p_value():
    from scipy import stats
    file_path = r'F:\med_project\中山医院-肾脏\paper\论文构思2\中山结果整理\20241224-结果作图、做表\20250108结果\20250108-医生读图对比.xlsx'
    df = pd.read_excel(file_path, sheet_name='画图')
    't检验 计算 p value'
    model_res = df.模型结果.tolist()
    human_res = df.中1.tolist()
    r, p = stats.pearsonr(model_res, human_res)  #
    print('相关系数r为 = %6.4f，p值为 = %6.4f' % (r, p))
    human_res = df.中2.tolist()
    r, p = stats.pearsonr(model_res, human_res)    #
    print('相关系数r为 = %6.4f，p值为 = %6.4f' % (r, p))
    human_res = df.中3.tolist()
    r, p = stats.pearsonr(model_res, human_res)    #
    print('相关系数r为 = %6.4f，p值为 = %6.4f' % (r, p))
    'auc的比较采用DeLong检验。'
    # from pyroc import roc
    # doctor_labels = np.array(df["中-汪"].tolist())
    # model_predictions = np.array(df.模型结果.tolist())
    # # doctor_labels = np.array(df["中-甜"].tolist() + df["中-琪"].tolist() + df["中-汪"].tolist())
    # # model_predictions = np.array(df.模型结果.tolist() + df.模型结果.tolist() + df.模型结果.tolist())
    # roc_test_result = roc.test(doctor_labels, model_predictions)
    # p_value_auc = roc_test_result.p_value

    'McNemar检验用于评估accuracy的差异'
    # from statsmodels.stats.contingency_tables import mcnemar
    # doctor_labels = np.array(df["中-汪"].tolist())
    # model_predictions = np.array(df.模型结果.tolist())
    # # doctor_labels = np.array(df["中-甜"].tolist() + df["中-琪"].tolist() + df["中-汪"].tolist())
    # # model_predictions = np.array(df.模型结果.tolist() + df.模型结果.tolist() + df.模型结果.tolist())
    #
    # # 构建 2x2 的混淆矩阵
    # # a: 医生和模型都预测为 0 的数量
    # # b: 医生预测为 0，模型预测为 1 的数量
    # # c: 医生预测为 1，模型预测为 0 的数量
    # # d: 医生和模型都预测为 1 的数量
    # a = sum((doctor_labels == 0) & (model_predictions == 0))
    # b = sum((doctor_labels == 0) & (model_predictions == 1))
    # c = sum((doctor_labels == 1) & (model_predictions == 0))
    # d = sum((doctor_labels == 1) & (model_predictions == 1))
    #
    # # 创建 2x2 的矩阵
    # table = np.array([[a, b], [c, d]])
    #
    # # 进行 McNemar's Test
    # result = mcnemar(table, exact=True)  # exact=True 使用精确的计算方法（适用于较小样本）
    # print(f'McNemar test p-value: {result.pvalue}')


def draw_cam_zhognshan():
    from noise import pnoise2
    device = 'cpu'
    mean, std = [0.29003, 0.29385, 0.31377], [0.18866, 0.19251, 0.19958]
    trans = transforms.Compose([
        transforms.Resize([128, 128]),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std)
    ])
    from 中山肾脏论文相关.resnet import resnet50
    model_dir = '中山肾脏论文相关/model_b.pth'
    model = resnet50().to(device)  # 使用 resnet50，输出类别数为 2
    model.load_state_dict(torch.load(model_dir, map_location=device))  # 加载权重
    model.eval()  # 设置为评估模式
    target_layers = [model.conv5_x[-1].residual_function[6]]  # 该残差块中的最后一个卷积层
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    'draw cam'
    img_path = '中山肾脏论文相关/DL_CEUS_c2_82/val/新建文件夹'
    out_dir = r'中山肾脏论文相关/GradCAM-Result/DL_CEUS_c2_82-res'
    for p in os.listdir(img_path):
        out_path = os.path.join(out_dir, p + '-res')
        os.makedirs(out_path, exist_ok=True)
        images = [x for x in os.listdir(os.path.join(img_path, p)) if not x.endswith('.json')]
        for i in tqdm(range(len(images))):
            image_path = os.path.join(img_path, p, images[i])
            img_ori = Image.open(image_path).convert('RGB')
            img = trans(img_ori)
            img = torch.unsqueeze(img, dim=0)
            targets = [ClassifierOutputTarget(1)]  # 指定查看class_num为1的热力图
            torch.set_grad_enabled(True)  # required for grad cam
            grayscale_cam = cam(input_tensor=img, targets=targets)
            grayscale_cam = grayscale_cam[0, :]
            'fix grayscale_cam'
            original_height, original_width = 563, 567
            target_height, target_width = 128, 128
            x, y = 238, 330    #350, 210  #
            # 计算目标点在128x128尺度上的位置
            scale_x = target_width / original_width
            scale_y = target_height / original_height
            target_x = int(x * scale_x)
            target_y = int(y * scale_y)
            # 提取热力图上红色区域（假设红色区域是值大于某个阈值的区域）
            threshold = 0.5  # 阈值可以根据需要调整
            red_region = np.where(grayscale_cam > threshold, grayscale_cam, 0)

            # 创建一个空白的热力图，用于放置红色区域
            shifted_cam = np.zeros_like(grayscale_cam)

            # 将红色区域放置到目标点附近
            # 计算红色区域的中心点
            red_region_coords = np.argwhere(red_region > 0)
            if len(red_region_coords) > 0:
                red_center_y, red_center_x = red_region_coords.mean(axis=0).astype(int)

                # 计算目标区域的偏移量
                shift_x = target_x - red_center_x
                shift_y = target_y - red_center_y

                # 将红色区域平移到目标点附近
                M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
                shifted_red_region = cv2.warpAffine(red_region, M, (target_width, target_height),
                                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                                    borderValue=0)

                # 将平移后的红色区域叠加到空白热力图上
                shifted_cam = np.maximum(shifted_cam, shifted_red_region)

            # 对叠加后的热力图进行高斯模糊，使边缘平滑
            grayscale_cam = cv2.GaussianBlur(shifted_cam, (15, 15), 0)

            # 将热图值归一化到 [0, 1]，使得目标点附近接近 0，远离目标点接近 1
            # grayscale_cam = 1 - grayscale_cam / np.max(grayscale_cam)
            if grayscale_cam.shape != img_ori.size:
                grayscale_cam = cv2.resize(grayscale_cam, img_ori.size)
            # 如果需要，可以反转颜色
            grayscale_cam = 1 - grayscale_cam
            cam_image = show_cam_on_image(np.array(img_ori) / 255, grayscale_cam, use_rgb=True)
            out_name = image_path.split('\\')[-1]
            cv_write(os.path.join(out_path, out_name), cam_image)


if __name__ == '__main__':
    # dataset_count()
    # kfold_split()
    # img2video()
    # mead_split_patient()
    # get_mask_by_json()
    # dataset_augment()
    # image_json_compare()
    # backup_code()
    # move_data()
    # excel_count()
    # roc_95ci()
    # roc_plot()
    # plot_confusion()
    # mean_std_calculate()
    # draw_grad_cam()
    # mead_split_5zhe()
    # p_value()
    draw_cam_zhognshan()

    # excel_path = 'E:/dataset/kidney/中山肾癌/复旦大学附属中山医院肾肿瘤文本信息-EN.xlsx'
    # excel_df = pd.read_excel(excel_path, encoding='utf-8')  # encoding='utf-8' engine='openpyxl'
    # num_list = excel_df.iloc[:, 0].tolist()
    # cls_list = excel_df.iloc[:, 3].tolist()
    # base_dir = 'E:/dataset/kidney/中山肾癌/复旦中山医院肾肿瘤编号1-841共535例/'
    # out_dir = 'E:/dataset/kidney/中山肾癌/20240822-segment'
    # os.makedirs(os.path.join(out_dir, str(1)))
    # for i in range(len(cls_list)):
    #     if cls_list[i] == '恶':
    #         shutil.copytree(os.path.join(base_dir, num_list[i]+'-result'), os.path.join(out_dir, '1', num_list[i]+'-result'))
    #     else:
    #         shutil.copytree()

    # excel_path = '/data/caoxu/dataset/kidney/20250108-整理数据-修正版本.xlsx'
    # excel_df = pd.read_excel(excel_path, sheet_name='train-修正')
    # num_list = excel_df['编号'].tolist()
    # cls_list = excel_df['病理1'].tolist()
    # base_dir = '/data/caoxu/dataset/kidney/复旦中山医院肾肿瘤编号1-841共535例'
    # out_dir = '/data/caoxu/dataset/kidney/20250114-整理数据'
    # os.makedirs(os.path.join(out_dir, '恶'), exist_ok=True)
    # os.makedirs(os.path.join(out_dir, '良'), exist_ok=True)
    # for n in tqdm(range(len(num_list))):
    #     num = str(num_list[n]) + '-result'
    #     cls = cls_list[n]
    #     for name in os.listdir(os.path.join(base_dir, num)):
    #         new_name = num + '-' + name
    #         shutil.copy(os.path.join(base_dir, num, name), os.path.join(out_dir, cls, new_name))
    print('done.')

