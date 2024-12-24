# import six
# import os
# import numpy as np
# import radiomics
import pandas as pd
# import SimpleITK as sitk
# from radiomics import featureextractor
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
# from tqdm import tqdm
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# def catch_features(image_path, mask_path):
#     if image_path is None or mask_path is None:  # Something went wrong, in this case PyRadiomics will also log an error
#         raise Exception('Error getting testcase!')  # Raise exception to prevent cells below from running in case of "run all"
#     settings = {}
#     settings['binWidth'] = 25  # 5
#     settings['sigma'] = [3, 5]
#     settings['Interpolator'] = sitk.sitkBSpline
#     settings['resampledPixelSpacing'] = [1, 1, 1]  # 3,3,3
#     settings['voxelArrayShift'] = 1000  # 300
#     settings['normalize'] = True
#     settings['normalizeScale'] = 100
#     extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
#     # extractor = featureextractor.RadiomicsFeatureExtractor()
#     # print('Extraction parameters:\n\t', extractor.settings)
#
#     extractor.enableImageTypeByName('LoG')
#     extractor.enableImageTypeByName('Wavelet')
#     extractor.enableAllFeatures()
#     extractor.enableFeaturesByName(
#         firstorder=['Energy', 'TotalEnergy', 'Entropy', 'Minimum', '10Percentile', '90Percentile', 'Maximum', 'Mean',
#                     'Median', 'InterquartileRange', 'Range', 'MeanAbsoluteDeviation', 'RobustMeanAbsoluteDeviation',
#                     'RootMeanSquared', 'StandardDeviation', 'Skewness', 'Kurtosis', 'Variance', 'Uniformity'])
#     extractor.enableFeaturesByName(
#         shape=['VoxelVolume', 'MeshVolume', 'SurfaceArea', 'SurfaceVolumeRatio', 'Compactness1', 'Compactness2',
#                'Sphericity', 'SphericalDisproportion', 'Maximum3DDiameter', 'Maximum2DDiameterSlice',
#                'Maximum2DDiameterColumn', 'Maximum2DDiameterRow', 'MajorAxisLength', 'MinorAxisLength',
#                'LeastAxisLength', 'Elongation', 'Flatness'])
#     # 上边两句我将一阶特征和形状特征中的默认禁用的特征都手动启用，为了之后特征筛选
#     # print('Enabled filters:\n\t', extractor.enabledImagetypes)
#     feature_cur = []
#     feature_name = []
#     result = extractor.execute(image_path, mask_path, label=255)
#     for key, value in six.iteritems(result):
#         # print('\t', key, ':', value)
#         feature_name.append(key)
#         feature_cur.append(value)
#     return feature_cur, feature_name


def radiomics_feature():
    # image_dir = 'D:/med_project/kidney_shiyuan/image'
    # all_features = []
    # all_labels = []
    # all_img = []
    # names = []
    # ids = []
    # with open('数据名称对应.txt', 'r', encoding='utf-8') as file:
    #     for line in file:
    #         ids.append(line.split(',')[0])
    #         names.append(line.split(',')[1].replace('\n', '').split('\\')[-1])
    # for root, dirs, files in os.walk(image_dir):
    #     for f in files:
    #         print(f)
    #         image_path = os.path.join(root, f)
    #         mask_path = os.path.join(root.replace('image', 'mask'), f)
    #         features, feature_names = catch_features(image_path, mask_path)
    #         all_features.append(features)
    #         label = 'malignant' if 'malignant' in root else 'benign'
    #         all_labels.append(label)
    #         idx = ids.index(f.split('.')[0])
    #         all_img.append(names[idx])

    # # 将特征和标签转换为DataFrame
    # df_features = pd.DataFrame(all_features, columns=feature_names)
    # df_labels = pd.DataFrame(all_labels, columns=['Label'])
    # all_img = pd.DataFrame(all_img, columns=['image_name'])
    # # 合并特征和标签
    # df_data = pd.concat([all_img, df_features, df_labels], axis=1)
    # # 保存到Excel
    # df_data.to_excel('radiomics_features.xlsx', index=False)

    # 读取excel 提取数据
    excel_data = pd.read_excel('radiomics_features111.xlsx')
    l = excel_data.shape[1] - 1
    df_features = excel_data.iloc[:, :l]
    df_labels = excel_data.Label

    # 准备训练和测试数据
    x_train, x_test, y_train, y_test = train_test_split(df_features, df_labels, test_size=0.2, random_state=0)

    # 特征选择：选择最重要的100个特征
    selector = SelectKBest(score_func=f_classif, k=100)
    x_train_selected = selector.fit_transform(x_train, y_train)
    x_test_selected = selector.transform(x_test)

    # 输出选中的特征名称
    selected_features = selector.get_support(indices=True)
    print("选中的特征名称:", df_features.columns[selected_features].tolist())
    
    # 数据标准化
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_selected)
    x_test_scaled = scaler.transform(x_test_selected)

    # 使用SVM进行分类
    classifier = SVC(kernel='linear', verbose=1)
    classifier.fit(x_train_scaled, y_train)

    # 预测和打印分类报告
    y_pred = classifier.predict(x_test_scaled)
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    radiomics_feature()
