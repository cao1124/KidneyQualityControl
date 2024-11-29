#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : clip_train.py
@Author  : cao xu
@Time    : 2024/4/19 13:54
clip model提取 img_feature和text_feature，concatenate后送入resnet分类
"""
import os
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
        print(self.images[idx], images.shape, caption, label)
        return images, caption, label


def load_data(data_path, num_list, excel_df, batch_size, preprocess):
    df = {'image': [], 'caption': [], 'label': []}
    'fold dataset'
    # num_list = excel_df.iloc[:, 0].tolist()
    # for f in image_path:
    #     for c in os.listdir(f):
    #         for p in os.listdir(os.path.join(f, c)):
    #             idx = num_list.index(int(p))
    #             for n in os.listdir(os.path.join(f, c, p)):
    #                 df['image'].append(os.path.join(f, c, p, n))
    #                 df['label'].append(int(c))
    for p_num in num_list:
        idx = num_list.index(int(p_num))
        for name in os.listdir(os.path.join(data_path, str(p_num) + '-result')):
            df['image'].append(os.path.join(data_path, str(p_num) + '-result', name))
            c = excel_df.iloc[idx][3]
            description = excel_df.iloc[idx][2]
            if c == '良':
                # cla = 'benign'
                df['label'].append(0)
            else:
                # cla = 'malignant'
                df['label'].append(1)
            if excel_df.iloc[idx][4] == '女':
                sex = 'woman'
            else:
                sex = 'man'
            year = int(excel_df.iloc[idx][5])
            # resect = int(excel_df.iloc[idx][6])
            loc = int(excel_df.iloc[idx][6])
            if loc == 0:
                located = 'left kidney'
            else:
                located = 'right kidney'
            pa = int(excel_df.iloc[idx][7])
            if pa == 0:
                part = 'upper kidney location'
            elif pa == 1:
                part = 'middle kidney location'
            else:
                part = 'lower kidney location'
            maximum = int(excel_df.iloc[idx][8])
            df['caption'].append(f"A photo of a kidney cancer image showing a tumor with {description} in "
                                 f"a {year}-year-old {sex}, with a maximum diameter of {maximum} mm, located "
                                 f"on the {located} kidney, in the {part}.")
            # df['caption'].append("A photo of a {}-year-old {} with kidney cancer.".format(year, sex))
            # df['caption'].append(
            #     "a photo of {} kidney cancer image in a {}-year-old {}.".format(cla, year, sex))

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


def train(num_epochs, batch_size, learning_rate, image_path, excel_df, save_path, device):
    # 加载CLIP模型
    model_clip, _ = load_pretrian_model('ViT-B/32', device)  # ViT-B/32
    for i in range(5):
        print('五折交叉验证 第{}次实验:'.format(i))
        # fold 数据划分
        # test_path = [os.path.join(image_path, 'fold4/')]
        # fold_list = ['fold0/', 'fold1/', 'fold2/', 'fold3/']
        # valid_path = [os.path.join(image_path, fold_list[3 - i])]
        # train_path = [os.path.join(image_path, x) for x in fold_list if x != fold_list[3 - i]]
        'csv 5fold dataset'
        train_list, valid_list, test_list = [], [], []
        num_list = excel_df.iloc[:, 0].tolist()
        temp = func(shuffle(num_list, random_state=i), int(len(num_list) * 0.2), m=5)
        for index, cross in enumerate(temp):
            if index == i:
                valid_list.append(cross)
            elif index == i + 1 and i < 4:
                test_list.append(cross)
            elif index == 0 and i == 4:
                test_list.append(cross)
            else:
                train_list.append(cross)
        # 加载模型  resnet
        model_classify = models.resnext50_32x4d(pretrained=True)
        model_classify.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
        # model_classify.conv1 = nn.Conv2d(1024, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # 加载模型  densenet
        # model_classify = models.densenet161(pretrained=True)
        # model_classify.classifier = nn.Linear(in_features=2208, out_features=2, bias=True)
        # model_classify.features.conv0 = nn.Conv2d(1024, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if torch.cuda.device_count() > 1:
            model_classify = nn.DataParallel(model_classify)
        model_classify.to(device)

        # 损失函数和优化器
        loss_func = nn.CrossEntropyLoss().to(device)
        optimizer = optim.NAdam(model_classify.parameters(), lr=learning_rate, betas=(0.8, 0.888), eps=1e-08,
                                weight_decay=2e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.005)

        # 加载数据集
        train_dataloader, train_size = load_data(image_path, train_list[0], excel_df, batch_size, image_transforms['train'])
        valid_dataloader, valid_size = load_data(image_path, valid_list[0], excel_df, batch_size, image_transforms['valid'])
        test_dataloader, test_size = load_data(image_path, test_list[0], excel_df, batch_size, image_transforms['valid'])
        print('train_size:{}, valid_size:{}, test_size:{}'.format(train_size, valid_size, test_size))
        # 训练
        fc_layer = torch.nn.Linear(1024, 3 * 224 * 224).to(device)
        best_test_acc, best_valid_acc, best_valid_recall, best_epoch = 0.0, 0.0, 0.0, 0
        history = []
        for epoch in range(num_epochs):
            print("Epoch: {}/{}".format(epoch + 1, num_epochs))
            train_loss, valid_loss, num_correct = 0.0, 0.0, 0

            'train'
            model_classify.train()
            torch.cuda.empty_cache()  # 释放缓存分配器当前持有的且未占用的缓存显存
            for step, batch in enumerate(train_dataloader):
                img = batch[0].to(device)
                text = clip.tokenize(batch[1]).to(device)
                label = batch[2].to(device)

                img_feature, text_feature = model_clip(img, text)
                img_with_text = torch.cat((img_feature, text_feature), dim=1).float()
                img_with_text = img_with_text.view(img_with_text.size(0), -1, 1, 1)
                # 使用一个全连接层将特征映射到一个更高维度的空间
                img_with_text_mapped = fc_layer(img_with_text.view(-1, 1024)).view(batch_size, 3, 224, 224)
                output = model_classify(img_with_text_mapped)

                loss_step = loss_func(output, label)
                train_loss += loss_step.item()
                temp_num_correct = torch.eq(output.argmax(dim=1), label).sum().float().item()
                num_correct += temp_num_correct

                optimizer.zero_grad()  # reset gradient
                loss_step.backward()
                optimizer.step()
                lr_scheduler.step()

            train_loss /= train_size
            train_acc = num_correct / train_size
            print("Train Epoch: {}, Loss: {:.4f}, Acc: {:.4f}".format(epoch + 1, train_loss, train_acc))

            'validation'
            num_correct = 0
            valid_true, valid_pred = [], []
            model_classify.eval()
            torch.cuda.empty_cache()  # 释放缓存分配器当前持有的且未占用的缓存显存
            with torch.no_grad():
                for step, batch in enumerate(valid_dataloader):
                    img = batch[0].to(device)
                    text = clip.tokenize(batch[1]).to(device)
                    label = batch[2].to(device)

                    img_feature, text_feature = model_clip(img, text)
                    img_with_text = torch.cat((img_feature, text_feature), dim=1).float()
                    img_with_text = img_with_text.view(img_with_text.size(0), -1, 1, 1)
                    # 使用一个全连接层将特征映射到一个更高维度的空间
                    img_with_text_mapped = fc_layer(img_with_text.view(-1, 1024)).view(batch_size, 3, 224, 224)
                    output = model_classify(img_with_text_mapped)

                    loss_step = loss_func(output, label)
                    valid_loss += loss_step.item()
                    num_correct += torch.eq(output.argmax(dim=1), label).sum().float().item()

                    valid_true.extend(label.cpu().numpy())
                    valid_pred.extend(output.argmax(dim=1).cpu().numpy())

            valid_loss /= valid_size
            valid_acc = num_correct / valid_size
            print("Val Epoch: {}, Loss: {:.4f}, Acc: {:.4f}".format(epoch + 1, valid_loss, valid_acc))
            'best acc save checkpoint'
            if best_valid_acc < valid_acc:
                print('confusion_matrix:\n{}'.format(confusion_matrix(valid_true, valid_pred)))
                print(
                    'classification_report:\n{}'.format(classification_report(valid_true, valid_pred, digits=4)))
                best_valid_acc = valid_acc
                best_epoch = epoch + 1
                torch.save(model_classify, os.path.join(save_path, 'fold' + str(i) + '-best-model.pt'))
            print("Epoch: {:03d}, Train Loss: {:.4f}, Acc: {:.4f}, Valid Loss: {:.4f}, Acc:{:.4f}"
                  .format(epoch + 1, train_loss, train_acc, valid_loss, valid_acc))
            print("validation best: {:.4f} at epoch {}".format(best_valid_acc, best_epoch))
            history.append([train_loss, valid_loss])
        history = np.array(history)
        plt.clf()  # 清图
        plt.plot(history)
        plt.legend(['Tr Loss', 'Val Loss'])
        plt.xlabel('Epoch Number')
        plt.ylabel('Loss')
        plt.ylim(0, np.max(history))
        plt.savefig(os.path.join(save_path, 'loss_curve' + str(i) + '.png'))

        'test'
        model_classify = torch.load(os.path.join(save_path, 'fold' + str(i) + '-best-model.pt'))
        num_correct = 0
        test_true, test_pred = [], []
        model_classify.eval()
        torch.cuda.empty_cache()  # 释放缓存分配器当前持有的且未占用的缓存显存
        with torch.no_grad():
            for step, batch in enumerate(test_dataloader):
                img = batch[0].to(device)
                text = clip.tokenize(batch[1]).to(device)
                label = batch[2].to(device)

                img_feature, text_feature = model_clip(img, text)
                img_with_text = torch.cat((img_feature, text_feature), dim=1).float()
                img_with_text = img_with_text.view(img_with_text.size(0), -1, 1, 1)
                # 使用一个全连接层将特征映射到一个更高维度的空间
                img_with_text_mapped = fc_layer(img_with_text.view(-1, 1024)).view(batch_size, 3, 224, 224)
                output = model_classify(img_with_text_mapped)

                num_correct += torch.eq(output.argmax(dim=1), label).sum().float().item()
                test_true.extend(label.cpu().numpy())
                test_pred.extend(output.argmax(dim=1).cpu().numpy())
            test_acc = num_correct / test_size
            print("test accuracy: {:.4f}".format(test_acc))
            print('confusion_matrix:\n{}'.format(confusion_matrix(test_true, test_pred)))
            print('classification_report:\n{}'.format(classification_report(test_true, test_pred, digits=4)))


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch = 500
    batch_size = 128
    learning_rate = 1e-3
    image_path = '/mnt/sdb/caoxu/kidney/复旦中山医院肾肿瘤编号1-841共535例'
    excel_path = '/mnt/sdb/caoxu/kidney/复旦大学附属中山医院肾肿瘤文本信息-EN.xlsx'
    excel_df = pd.read_excel(excel_path, encoding='utf-8')  # encoding='utf-8' engine='openpyxl'
    save_path = 'res/20240821-clip-resnext50-classify'
    os.makedirs(save_path, exist_ok=True)
    train(epoch, batch_size, learning_rate, image_path, excel_df, save_path, device)


if __name__ == '__main__':
    main()
