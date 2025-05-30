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


def roc_plot():
    print('roc_plot')

    # 读取数据（与confusion_matrix相同的Excel文件）
    excel_path = r'E:\med_project\中山医院-肾脏\中山结果整理\20250317\20250320-结果整理.xlsx'
    excel_data = pd.read_excel(excel_path, sheet_name='域外测试结果')

    # 获取真实标签和预测概率（假设有'Prob'列存储模型预测概率）
    labels = excel_data['Label'].tolist()
    probs = excel_data['Prob'].tolist()  # 模型预测的概率值（需要0-1之间的值）

    # 计算ROC曲线和AUC
    fpr, tpr, thresholds = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    # 绘制ROC曲线
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.plot([0, 1], [0, 1], color='m', linestyle='--')
    plt.rcParams.update({'font.size': 8})
    plt.legend(loc='lower right')
    plt.xlabel("1-Specificity", fontsize=15)
    plt.ylabel("Sensitivity", fontsize=15)
    plt.title("ROC")
    plt.draw()
    plt.savefig('20250320-模型和医生读图对比ROC图-域外测试数据.tiff')


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
    excel_path = r'E:\med_project\中山医院-肾脏\中山结果整理\20250317\20250320-结果整理.xlsx'
    excel_data = pd.read_excel(excel_path, sheet_name='域外测试结果')
    labels = excel_data['Label'].tolist()
    title = 'Model-UT'      # Model-G Model-GC Early  Late  MultiHead   Model-UT 域内测试结果 域外测试结果
    pres = excel_data[title].tolist()
    # labels = label + label + label
    # pres1 = excel_data['中-甜'].tolist()
    # pres2 = excel_data['中-琪'].tolist()
    # pres3 = excel_data['中-汪'].tolist()
    # pres = pres1 + pres2 + pres3

    save_name = f'混淆矩阵图-{title} 域外测试结果 分类结果.tiff'
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
    confusion_matrix_plot(cm_normalized, cls, title='Confusion Matrices of Specific Diagnoses', xtitle=title, cmap=plt.cm.Blues)

    plt.draw()
    plt.savefig(save_name)


if __name__ == '__main__':
    # roc_plot()
    plot_confusion()

    print('done.')

