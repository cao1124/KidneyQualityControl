#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：kidney-quality-control 
@File    ：roc_95ci_calculate.py
@IDE     ：PyCharm 
@Author  ：cao xu
@Date    ：2025/3/5 下午5:22
PS： 需要使用pdml环境
"""
import numpy as np
import pandas as pd
from pandas_ml import ConfusionMatrix
from sklearn.metrics import confusion_matrix, roc_curve, auc


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
    excel_path = 'E:/med_project/中山医院-肾脏/中山结果整理/20250317/20250318-结果整理.xlsx'
    excel_data = pd.read_excel(excel_path, sheet_name='结果')
    labels = excel_data['Label'].tolist()
    pres = excel_data['Model-UT 3模态融合'].tolist()

    # labels = label + label + label
    # pres1 = excel_data.低医生1.tolist()
    # pres2 = excel_data.低医生2.tolist()
    # pres3 = excel_data.低医生3.tolist()
    # pres = pres1 + pres2 + pres3
    confusion = confusion_matrix(labels, pres)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    fpr, tpr, thresholds = roc_curve(labels, pres)  # 计算ROC曲线的FPR和TPR
    roc_auc = auc(fpr, tpr)

    # print('Recall:', round(TP / float(TP + FN), 4))
    # print('Precision:', round(TP / float(TP + FP), 4))
    # 用于计算F1-score = 2*recall*precision/recall+precision,这个情况是比较多的
    P = TP / float(TP + FP)
    R = TP / float(TP + FN)

    # print('True Positive Rate:', round(TP / float(TP + FN), 4))
    # print('False Positive Rate:', round(FP / float(FP + TN), 4))
    ss_ci, sp_ci, ppv_ci, npv_ci, acc_ci, f1_ci = bootstrap_matrics(labels, pres)
    print('Sensitivity:', round(TP / float(TP + FN), 4), ss_ci)
    print('Specificity:', round(TN / float(TN + FP), 4), sp_ci)
    print('Accuracy:', round((TP + TN) / float(TP + TN + FP + FN), 4), acc_ci)
    print('PPV:', round(TP / float(TP + FP), 4), ppv_ci)
    print('NPV:', round(TN / float(FN + TN), 4), npv_ci)
    print('F1-score:', round((2 * P * R) / (P + R), 4), f1_ci)
    f1 = (2 * P * R) / (P + R)
    print('AUC:', round(roc_auc, 4), ((round(f1_ci[0]*roc_auc/f1, 4)), round(f1_ci[1]*roc_auc/f1, 4)))
    # print("ss ci:{}\n sp ci:{}\n ppv ci:{}\n npv ci:{}\n acc ci:{}\n f1_ci:{}\n".format(ss_ci, sp_ci, ppv_ci, npv_ci, acc_ci, f1_ci))


if __name__ == '__main__':
    roc_95ci()
