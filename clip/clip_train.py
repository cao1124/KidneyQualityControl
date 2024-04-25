#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : clip_train.py
@Author  : cao xu
@Time    : 2024/4/19 13:54
"""
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
import os
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
        return images, caption, label


def load_data(image_path, excel_df, batch_size, preprocess):
    df = {'image': [], 'caption': [], 'label': []}
    num_list = excel_df.iloc[:, 0].tolist()
    for f in image_path:
        for c in os.listdir(f):
            for p in os.listdir(os.path.join(f, c)):
                idx = num_list.index(int(p))
                for n in os.listdir(os.path.join(f, c, p)):
                    df['image'].append(os.path.join(f, c, p, n))
                    df['label'].append(int(c))
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
    for i in range(4):
        print('五折交叉验证 第{}次实验:'.format(i))
        # 数据划分
        test_path = [os.path.join(image_path, 'fold4/')]
        fold_list = ['fold0/', 'fold1/', 'fold2/', 'fold3/']
        valid_path = [os.path.join(image_path, fold_list[3 - i])]
        train_path = [os.path.join(image_path, x) for x in fold_list if x != fold_list[3 - i]]

        # 加载模型
        model_classify = models.resnet50(pretrained=True)
        model_classify.conv1 = nn.Conv2d(1024, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model_classify.fc = nn.Linear(in_features=2048, out_features=2, bias=True)
        if torch.cuda.device_count() > 1:
            model_classify = nn.DataParallel(model_classify)
        model_classify.to(device)

        # 损失函数和优化器
        loss_func = nn.CrossEntropyLoss().to(device)
        optimizer = optim.NAdam(model_classify.parameters(), lr=learning_rate, betas=(0.8, 0.888), eps=1e-08,
                                weight_decay=2e-4)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.005)

        # 加载数据集
        train_dataloader, train_size = load_data(train_path, excel_df, batch_size, image_transforms['train'])
        valid_dataloader, valid_size = load_data(valid_path, excel_df, batch_size, image_transforms['valid'])
        test_dataloader, test_size = load_data(test_path, excel_df, batch_size, image_transforms['valid'])
        print('train_size:{}, valid_size:{}, test_size:{}'.format(train_size, valid_size, test_size))
        # 训练
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
                output = model_classify(img_with_text)

                loss_step = loss_func(output, label)
                train_loss += loss_step.item()
                temp_num_correct = torch.eq(output.argmax(dim=1), label).sum().float().item()
                num_correct += temp_num_correct

                optimizer.zero_grad()  # reset gradient
                loss_step.backward()
                optimizer.step()
                lr_scheduler.step()

            train_loss /= len(train_dataloader)
            train_acc = num_correct / len(train_dataloader)
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
                    output = model_classify(img_with_text)

                    loss_step = loss_func(output, label)
                    valid_loss += loss_step.item()
                    num_correct += torch.eq(output.argmax(dim=1), label).sum().float().item()

                    valid_true.extend(label.cpu().numpy())
                    valid_pred.extend(output.argmax(dim=1).cpu().numpy())

            valid_loss /= len(valid_dataloader)
            valid_acc = num_correct / len(valid_dataloader)
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
                output = model_classify(img_with_text)

                num_correct += torch.eq(output.argmax(dim=1), label).sum().float().item()
                test_true.extend(label.cpu().numpy())
                test_pred.extend(output.argmax(dim=1).cpu().numpy())
            test_acc = num_correct / len(test_dataloader)
            print("test accuracy: {:.4f}".format(test_acc))
            print('confusion_matrix:\n{}'.format(confusion_matrix(test_true, test_pred)))
            print('classification_report:\n{}'.format(classification_report(test_true, test_pred, digits=4)))


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epoch = 200
    batch_size = 256
    learning_rate = 1e-3
    image_path = '/media/user/Disk1/caoxu/dataset/kidney/zhongshan/20240312-kidney-5fold'
    excel_path = '复旦中山医院肾肿瘤病理编号1-600共508例.csv'
    excel_df = pd.read_csv(excel_path, encoding='utf-8')  # encoding='utf-8' engine='openpyxl'
    save_path = '20240425-clip-resnet50-classify'
    os.makedirs(save_path, exist_ok=True)
    train(epoch, batch_size, learning_rate, image_path, excel_df, save_path, device)


if __name__ == '__main__':
    main()
