#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : clip_train.py
@Author  : cao xu
@Time    : 2024/4/19 13:54
"""
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
import clip
import clip_model
from torch import nn, optim
import pandas as pd
from PIL import Image
import os
import warnings
matplotlib.use('AGG')
torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore")


class image_caption_dataset(Dataset):
    def __init__(self, df, preprocess):
        self.images = df["image"]
        self.caption = df["caption"]
        self.preprocess = preprocess

    def __len__(self):
        return len(self.caption)

    def __getitem__(self, idx):
        images = self.preprocess(Image.open(self.images[idx]))
        caption = self.caption[idx]
        return images, caption


def load_data(image_path, excel_df, batch_size, preprocess):
    df = {'image': [], 'caption': []}
    num_list = excel_df.iloc[:, 0].tolist()
    for f in os.listdir(image_path):
        if f != 'fold4':
            for c in os.listdir(os.path.join(image_path, f)):
                for p in os.listdir(os.path.join(image_path, f, c)):
                    idx = num_list.index(int(p))
                    for n in os.listdir(os.path.join(image_path, f, c, p)):
                        df['image'].append(os.path.join(image_path, f, c, p, n))
                        # cla = excel_df.iloc[idx][1]
                        # cla_type = excel_df.iloc[idx][2]
                        # sex = excel_df.iloc[idx][3]
                        if c == '0':
                            cla = 'benign'
                        else:
                            cla = 'malignant'
                        if excel_df.iloc[idx][3] == '女':
                            sex = 'woman'
                        else:
                            sex = 'man'
                        year = int(excel_df.iloc[idx][4])
                        df['caption'].append(
                            "a photo of {} kidney cancer image in a {}-year-old {}.".format(cla, year, sex))

    dataset = image_caption_dataset(df, preprocess)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_dataloader


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


def train(epoch, batch_size, learning_rate, image_path, excel_df, save_path, device):
    # 加载模型
    model, preprocess = load_pretrian_model('ViT-B/32', device)   # ViT-B/32

    # 加载数据集
    train_dataloader = load_data(image_path, excel_df, batch_size, preprocess)

    # 设置参数
    loss_img = nn.CrossEntropyLoss().to(device)
    loss_txt = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)
    history = []
    for i in range(epoch):
        total_loss = 0
        for images, texts in train_dataloader:
            texts = clip.tokenize(texts).to(device)
            images = images.to(device)

            logits_per_image, logits_per_text = model(images, texts)
            ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)

            # 反向传播
            cur_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

            optimizer.zero_grad()
            cur_loss.backward()
            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip_model.convert_weights(model)
            total_loss += cur_loss.cpu().item()
        print('epoch [%d] loss: %.3f' % (i + 1, total_loss))
        if total_loss != 'nan':
            history.append(total_loss)
    torch.save(model, os.path.join(save_path, '20240423-clip-classify-model.pt'))
    history = np.array(history)
    plt.clf()  # 清图
    plt.plot(history)
    plt.legend('Tr Loss')
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.ylim(0, np.max(history))
    plt.savefig(os.path.join(save_path, 'loss_curve.png'))


def main():
    epoch = 200
    batch_size = 16
    learning_rate = 1e-4
    image_path = '/home/ai999/dataset/kidney/20240312-kidney-5fold'
    excel_path = '复旦中山医院肾肿瘤病理编号1-600共508例.csv'
    excel_df = pd.read_csv(excel_path, encoding='utf-8')  # encoding='utf-8' engine='openpyxl'
    save_path = '20240425-clip-model-ViT-B-32-lr1e2'
    os.makedirs(save_path, exist_ok=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(epoch, batch_size, learning_rate, image_path, excel_df, save_path, device)


if __name__ == '__main__':
    main()
