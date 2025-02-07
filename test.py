#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：kidney-quality-control 
@File    ：test.py
@IDE     ：PyCharm 
@Author  ：cao xu
@Date    ：2025/2/5 下午4:36 
"""
import numpy as np
import pandas as pd
from pandas_ml import ConfusionMatrix
from sklearn.metrics import confusion_matrix


def bootstrap_matrics(labels, preds, nsamples=100):
    ss_values = []
    sp_values = []
    ppv_values = []
    npv_values = []
    acc_values = []
    F1_score_values = []

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
    excel_path = r'C:\Users\Administrator\Desktop\结果-20250205.xlsx'
    excel_data = pd.read_excel(excel_path)
    label = excel_data.label.tolist()
    # pres = excel_data.融合.tolist()
    labels = label + label + label
    pres1 = excel_data.高医生1.tolist()
    pres2 = excel_data.高医生2.tolist()
    pres3 = excel_data.高医生3.tolist()
    pres = pres1 + pres2 + pres3
    confusion = confusion_matrix(labels, pres)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    print('Accuracy:', round((TP + TN) / float(TP + TN + FP + FN), 4))
    print('Sensitivity:', round(TP / float(TP + FN), 4))
    print('Specificity:', round(TN / float(TN + FP), 4))
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


def p_value():
    from scipy import stats
    file_path = '20250205-中山反馈结果.xlsx'
    df = pd.read_excel(file_path, sheet_name='预测结果汇总')
    't检验 计算 p value'
    # model_res = df.低医生3.tolist()
    # human_res = df.高医生1.tolist()
    human_res = df.高医生3.tolist() + df.高医生3.tolist() + df.高医生3.tolist()
    model_res = df.低医生1.tolist() + df.低医生2.tolist() + df.低医生3.tolist()
    # human_res = df.高医生1.tolist() + df.高医生2.tolist() + df.高医生3.tolist()
    r, p = stats.pearsonr(model_res, human_res)  #
    print('相关系数r为 = %6.4f，p值为 = %6.4f' % (r, p))
    # human_res = df.高医生2.tolist()
    # r, p = stats.pearsonr(model_res, human_res)    #
    # print('相关系数r为 = %6.4f，p值为 = %6.4f' % (r, p))
    # human_res = df.高医生3.tolist()
    # r, p = stats.pearsonr(model_res, human_res)    #
    # print('相关系数r为 = %6.4f，p值为 = %6.4f' % (r, p))
    'auc的比较采用DeLong检验。'
    # from pyroc import roc
    # doctor_labels = np.array(df["中-汪"].tolist())
    # model_predictions = np.array(df.模型结果.tolist())
    # # doctor_labels = np.array(df["中-甜"].tolist() + df["中-琪"].tolist() + df["中-汪"].tolist())
    # # model_predictions = np.array(df.模型结果.tolist() + df.模型结果.tolist() + df.模型结果.tolist())
    # roc_test_result = roc.test(doctor_labels, model_predictions)
    # p_value_auc = roc_test_result.p_value

    'McNemar检验用于评估accuracy的差异'
    from statsmodels.stats.contingency_tables import mcnemar
    doctor_labels = np.array(model_res)
    model_predictions = np.array(human_res)
    # doctor_labels = np.array(df["高医生3"].tolist())
    # model_predictions = np.array(df.融合.tolist())
    # doctor_labels = np.array(df["中-甜"].tolist() + df["中-琪"].tolist() + df["中-汪"].tolist())
    # model_predictions = np.array(df.模型结果.tolist() + df.模型结果.tolist() + df.模型结果.tolist())

    # 构建 2x2 的混淆矩阵
    # a: 医生和模型都预测为 0 的数量
    # b: 医生预测为 0，模型预测为 1 的数量
    # c: 医生预测为 1，模型预测为 0 的数量
    # d: 医生和模型都预测为 1 的数量
    a = sum((doctor_labels == 0) & (model_predictions == 0))
    b = sum((doctor_labels == 0) & (model_predictions == 1))
    c = sum((doctor_labels == 1) & (model_predictions == 0))
    d = sum((doctor_labels == 1) & (model_predictions == 1))

    # 创建 2x2 的矩阵
    table = np.array([[a, b], [c, d]])

    # 进行 McNemar's Test
    result = mcnemar(table, exact=True)  # exact=True 使用精确的计算方法（适用于较小样本）
    print(f'McNemar test p-value: {result.pvalue}')


if __name__ == '__main__':
    # roc_95ci()
    p_value()
    print('done.')
