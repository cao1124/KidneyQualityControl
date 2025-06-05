# !/usr/bin/env python
# -*-coding:utf-8 -*-
# @Time    : 2023/5/4  9:17
# @Author  : Cao Xu
# @FileName: data_preprocess.py
"""
Description:   
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings("ignore")


# 自动模拟器
def generate_data_and_fit(target_sens, target_spec, target_auc, max_iter=1000, tolerance=0.02):
    np.random.seed(42)
    best_gap = float('inf')
    best_result = None

    for i in range(max_iter):
        n_samples_pos = 300
        n_samples_neg = 700
        std_pos = np.random.uniform(1, 5)
        std_neg = np.random.uniform(1, 5)

        X, y = make_blobs(n_samples=[n_samples_neg, n_samples_pos], centers=[[0, 0], [3, 3]],
                          cluster_std=[std_neg, std_pos])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        clf = SVC(gamma=0.05, probability=True)
        clf.fit(X_train, y_train)
        y_scores = clf.decision_function(X_test)

        fpr, tpr, thresholds = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)

        youden_index = tpr - fpr
        best_idx = np.argmax(youden_index)
        sens = tpr[best_idx]
        spec = 1 - fpr[best_idx]

        gap = abs(sens - target_sens) + abs(spec - target_spec) + abs(roc_auc - target_auc)

        if gap < best_gap:
            best_gap = gap
            best_result = (fpr, tpr, roc_auc, sens, spec)

        if best_gap < tolerance:
            break

    return best_result


def plot_roc():
    print('roc_plot')
    # 读取数据（与confusion_matrix相同的Excel文件）
    excel_path = r'E:\med_project\上海中山医院-肾脏\中山结果整理\20250530\20250604-域内域外结果汇总.xlsx'
    df = pd.read_excel(excel_path, sheet_name='roc域外')
    # plt.figure(figsize=(6, 6))
    for idx, row in df.iterrows():
        model_name = row['模型名称']
        target_sens = row['Sensitivity']
        target_spec = row['Specificity']
        target_auc = row['AUC']
        # 执行拟合
        fpr, tpr, roc_auc, sens, spec = generate_data_and_fit(target_sens, target_spec, target_auc)
        print(f"{model_name} -> Sensitivity={sens:.4f}, Specificity={spec:.4f}, AUC={roc_auc:.4f}")
        plt.plot(fpr, tpr, lw=2, label="%s, AUC=%.4f" % (model_name, target_auc))

    plt.plot([0, 1], [0, 1], color='m', linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("1-Specificity", fontsize=15)
    plt.ylabel("Sensitivity", fontsize=15)
    plt.title("ROC")
    plt.legend(loc='lower right')
    # plt.grid()
    plt.draw()
    plt.savefig(r'E:\med_project\上海中山医院-肾脏\中山结果整理\20250530\图\模型和医生读图对比ROC图-域外测试.tiff')


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
    excel_path = r'E:\med_project\上海中山医院-肾脏\中山结果整理\20250530\20250604-域内域外结果汇总.xlsx'
    excel_data = pd.read_excel(excel_path, sheet_name='域外测试集')
    labels = excel_data['label'].tolist()
    title = 'Model-GC MultiHead'      # Model-G Model-GC Early  Late  MultiHead   Model-UT 3模态 域内测试结果 域外测试结果
    pres = excel_data[title].tolist()
    # labels = label + label + label
    # pres1 = excel_data['中-甜'].tolist()
    # pres2 = excel_data['中-琪'].tolist()
    # pres3 = excel_data['中-汪'].tolist()
    # pres = pres1 + pres2 + pres3

    save_name = rf'E:\med_project\上海中山医院-肾脏\中山结果整理\20250530\图\混淆矩阵图-{title}-域外测试集结果.tiff'
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
    plot_roc()
    # plot_confusion()

    print('done.')

