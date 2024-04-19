#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : clip_train.py
@Author  : cao xu
@Time    : 2024/4/19 13:54
"""
from torch.utils.data import Dataset, DataLoader
import torch
import clip
import clip_model
from torch import nn, optim
import pandas as pd
from PIL import Image
import os


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
    for f in image_path:
        if f != 'fold4':
            for c in os.listdir(f):
                for p in os.listdir(os.path.join(f, c)):
                    idx = num_list.index(int(p))
                    for n in os.listdir(os.path.join(f, c, p)):
                        df['image'].append(os.path.join(f, c, p, n))
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
                        df['caption'].append("a photo of {} kidney cancer image in a {}-year-old {}.".format(cla, year, sex))

    dataset = image_caption_dataset(df, preprocess)
    train_dataloader = DataLoader(dataset, batch_size=batch_size)
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
    model, preprocess = load_pretrian_model('ViT-B/32', device)

    # 加载数据集
    train_dataloader = load_data(image_path, excel_df, batch_size, preprocess)

    # 设置参数
    loss_img = nn.CrossEntropyLoss().to(device)
    loss_txt = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

    for i in range(epoch):
        total_loss = 0
        for images, texts in train_dataloader:
            texts = clip.tokenize(texts).to(device)
            images = images.to(device)

            logits_per_image, logits_per_text = model(images, texts)
            ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)

            # 反向传播
            cur_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            total_loss += cur_loss
            optimizer.zero_grad()
            total_loss.backward()
            if device == "cpu":
                optimizer.step()
            else:
                convert_models_to_fp32(model)
                optimizer.step()
                clip_model.convert_weights(model)

        print('epoch [%d] loss: %.3f' % (i + 1, total_loss))
    torch.save(model, os.path.join(save_path, '20240419-clip-classify-model.pt'))


def main():
    epoch = 100
    batch_size = 16
    learning_rate = 5e-5
    image_path = '/home/ai999/dataset/kidney/20240312-kidney-5fold'
    excel_path = '/home/ai999/dataset/kidney/复旦中山医院肾肿瘤病理编号1-600共508例.xlsx'
    excel_df = pd.read_excel(excel_path, engine='openpyxl')  # , encoding='utf-8'
    save_path = 'clip-model'
    os.makedirs(save_path, exist_ok=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(epoch, batch_size, learning_rate, image_path, excel_df, save_path, device)


if __name__ == '__main__':
    main()
