#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：kidney-quality-control 
@File    ：p_value.py
@IDE     ：PyCharm 
@Author  ：cao xu
@Date    ：2025/3/6 下午1:51 
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_rel
from statsmodels.stats.contingency_tables import mcnemar
from scipy import stats


def compute_midrank(x):
    """Computes midranks."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float)
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx/m + sy/n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return 2 * (1 - stats.norm.cdf(z))


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    # DeLong检验实现代码（源自：https://github.com/yandexdataschool/roc_comparison）
    order = (-ground_truth).argsort()
    label_1_count = np.sum(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov)[0][0]


def p_value():
    file_path = r'F:\med_project\中山医院-肾脏\中山结果整理\20250305作图\20250305-结果整理.xlsx'
    df = pd.read_excel(file_path, sheet_name='内部测试集结果Model-G')
    model_res = np.array(df['Model-G'].tolist())
    human_res = np.array(df['Model-UT'].tolist())
    true_labels = np.array(df.病理诊断1.tolist())

    't检验 计算 p value'
    t_stat, p_val_t = ttest_rel(model_res, human_res)

    'auc的比较采用DeLong检验  模型结果和医生读图都是预测概率'
    p_val_delong = delong_roc_test(true_labels, model_res, human_res)

    'McNemar检验用于评估accuracy的差异'
    contingency_table = np.array([
        [np.sum((model_res == true_labels) & (human_res == true_labels)),  # 两者都正确
         np.sum((model_res == true_labels) & (human_res != true_labels))],  # 模型对医生错
        [np.sum((model_res != true_labels) & (human_res == true_labels)),  # 模型错医生对
         np.sum((model_res != true_labels) & (human_res != true_labels))]  # 两者都错
    ])

    # 执行McNemar检验
    mcnemar_result = mcnemar(contingency_table, exact=False)
    p_val_mcnemar = mcnemar_result.pvalue
    print(f"T检验p值: {p_val_t:.4f}")
    print(f"DeLong检验p值: {p_val_delong:.4f}")
    print(f"McNemar检验p值: {p_val_mcnemar:.4f}")


if __name__ == '__main__':
    p_value()
