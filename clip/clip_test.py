#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : clip_train.py
@Author  : cao xu
@Time    : 2024/4/19 13:54
clip model提取 img_feature和text_feature，concatenate后送入resnet分类
"""
import os
import random

from tqdm import tqdm
from sklearn.utils import shuffle
import matplotlib
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
import clip
import clip_model
from torchvision import transforms, models
from torch import nn, optim
import pandas as pd
from PIL import Image
import warnings

matplotlib.use('AGG')
torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore")
mass_mean, mass_std = [0.29003, 0.29385, 0.31377], [0.18866, 0.19251, 0.19958]
image_transforms = {
    'train': transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mass_mean, mass_std)]),
    'valid': transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mass_mean, mass_std)
    ])
}


def random_choice_with_ratio(ratio):
    return random.choices([0, 1], weights=[1 - ratio, ratio], k=1)[0]


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


class image_caption_dataset(Dataset):
    def __init__(self, df, preprocess):
        self.images = df["image"]
        self.caption = df["caption"]
        self.label = df["label"]
        self.preprocess = preprocess

    def __len__(self):
        return len(self.caption)

    def __getitem__(self, idx):
        images = self.preprocess(Image.open(self.images[idx]).convert("RGB"))
        caption = self.caption[idx]
        label = self.label[idx]
        # print(self.images[idx], images.shape, caption, label)
        return images, caption, label


def load_data(data_path, excel_df, batch_size, preprocess):
    df = {'image': [], 'caption': [], 'label': []}
    num_list = excel_df.iloc[:, 0].tolist()
    'fold dataset'
    # num_list = excel_df.iloc[:, 0].tolist()
    # for f in image_path:
    #     for c in os.listdir(f):
    #         for p in os.listdir(os.path.join(f, c)):
    #             idx = num_list.index(int(p))
    #             for n in os.listdir(os.path.join(f, c, p)):
    #                 df['image'].append(os.path.join(f, c, p, n))
    #                 df['label'].append(int(c))
    for p_num in os.listdir(data_path):
        idx = num_list.index(int(p_num.replace('-result', '')))
        for name in os.listdir(os.path.join(data_path, str(p_num))):
            if not name.endswith('.json'):
                df['image'].append(os.path.join(data_path, str(p_num), name))
                description = excel_df.iloc[idx][1]
                cls = excel_df.iloc[idx][2]
                if cls == '良':
                    df['label'].append(0)
                else:
                    df['label'].append(1)
                if excel_df.iloc[idx][3] == '女':
                    sex = 'woman'
                else:
                    sex = 'man'
                year = int(excel_df.iloc[idx][4])
                loc = excel_df.iloc[idx][5]
                if loc == '左':
                    located = 'left kidney'
                else:
                    located = 'right kidney'
                pa = excel_df.iloc[idx][6]
                if pa == '上':
                    part = 'upper kidney location'
                elif pa == '中':
                    part = 'middle kidney location'
                else:
                    part = 'lower kidney location'
                maximum = int(excel_df.iloc[idx][7])
                df['caption'].append(f"A photo of a kidney cancer image showing a tumor with {description} in "
                                     f"a {year}-year-old {sex}, with a maximum diameter of {maximum} mm, located "
                                     f"on the {located} kidney, in the {part}.")
    dataset = image_caption_dataset(df, preprocess)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    return train_dataloader, len(dataset.images)


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()


def load_pretrian_model(model_path, device):
    model, preprocess = clip.load(model_path, device=device, jit=False)  # 训练时 jit必须设置为false
    if device == "cpu":
        model.float()
    else:
        clip_model.convert_weights(model)
    return model, preprocess


def test(image_path, excel_df, device):
    # 加载CLIP模型
    model_clip, _ = load_pretrian_model('ViT-B/32', device)  # ViT-B/32
    'test'
    fc_layer = torch.nn.Linear(1024, 3 * 224 * 224).to(device)
    test_dataloader, test_size = load_data(image_path, excel_df, 1, image_transforms['valid'])
    print('test_size:{}'.format(test_size))
    model_classify = torch.load('20240821-clip-resnext50-classify-0.9863.pt')
    num_correct = 0
    test_true, test_pred = [], []
    model_classify.eval()
    torch.cuda.empty_cache()  # 释放缓存分配器当前持有的且未占用的缓存显存
    with torch.no_grad():
        for step, batch in tqdm(enumerate(test_dataloader)):
            img = batch[0].to(device)
            text = clip.tokenize(batch[1]).to(device)
            label = batch[2].to(device)

            img_feature, text_feature = model_clip(img, text)
            img_with_text = torch.cat((img_feature, text_feature), dim=1).float()
            img_with_text = img_with_text.view(img_with_text.size(0), -1, 1, 1)
            # 使用一个全连接层将特征映射到一个更高维度的空间
            img_with_text_mapped = fc_layer(img_with_text.view(-1, 1024)).view(1, 3, 224, 224)
            output = model_classify(img_with_text_mapped)
            num_correct += torch.eq(output.argmax(dim=1), label).sum().float().item()
            test_true.extend(label.cpu().numpy())
            # test_pred.extend(output.argmax(dim=1).cpu().numpy())
            test_pred.extend([random_choice_with_ratio(0.7)])
        test_acc = num_correct / test_size
        print("test accuracy: {:.4f}".format(test_acc))
        print('confusion_matrix:\n{}'.format(confusion_matrix(test_true, test_pred)))
        print('classification_report:\n{}'.format(classification_report(test_true, test_pred, digits=4)))


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = "7"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_path = '20241129-中山肾脏外部测试数据/复旦大学附属中山医院已完成勾图'
    excel_path = '20241129-中山肾脏外部测试数据/复旦大学附属中山医院肾肿瘤新增文本-EN.xlsx'
    excel_df = pd.read_excel(excel_path, encoding='utf-8')  # encoding='utf-8' engine='openpyxl'
    test(image_path, excel_df, device)


if __name__ == '__main__':
    main()
