import math
import os

import cv2
from PIL import Image
import numpy as np
import torch
import albumentations as albu
from pretrainedmodels.models import senet
from torch import nn, optim
from torch.utils.data import Dataset
from torchvision import transforms, models
from warmup_scheduler import GradualWarmupScheduler
from CBAM_ResNet import CBAM_Resnext, resnet18_cbam, resnet34_cbam, resnet50_cbam, resnet101_cbam, resnet152_cbam
from ECA_ResNet import eca_resnet50, eca_resNeXt50_32x4d

model_dict = {
    'efficientnet_b0': models.efficientnet_b0,
    'vgg16': models.vgg16,
    'vgg11': models.vgg11,
    "resnet18": models.resnet18,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
    "resnext50": models.resnext50_32x4d,
    "resnext101": models.resnext101_32x8d,
    "wide_resnet50_2": models.wide_resnet50_2,
    "wide_resnet101_2": models.wide_resnet101_2,
    'densenet121': models.densenet121,
    'densenet161': models.densenet161,
    'densenet169': models.densenet169,
    'densenet201': models.densenet201,
    "regnet_y_32gf": models.regnet_y_32gf,
    "SENet_resnet50": senet.se_resnet50,
    "SENet_resnext50": senet.se_resnext50_32x4d,
    "eca_resnet50": eca_resnet50,
    "eca_resnext50": eca_resNeXt50_32x4d,
    "resnet18_cbam": resnet18_cbam,
    "resnet34_cbam": resnet34_cbam,
    "resnet50_cbam": resnet50_cbam,
    "resnet101_cbam": resnet101_cbam,
    "resnet152_cbam": resnet152_cbam,
    "resnext50_cbam": CBAM_Resnext,
    "resnext101_cbam": CBAM_Resnext,
    "resnext152_cbam": CBAM_Resnext,
}

mass_mean, mass_std = [0.29003, 0.29385, 0.31377], [0.18866, 0.19251, 0.19958]

# 数据增强
image_transforms = {
    'train': transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        # transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
        # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 0.3), value=0, inplace=False),
        transforms.ToTensor(),
        transforms.Normalize(mass_mean, mass_std)]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([224, 224]),
        transforms.Normalize(mass_mean, mass_std)
    ])
}


def augment_compose(prob):
    train_transform = [
        albu.OneOf([albu.Rotate(p=prob, limit=180, border_mode=cv2.BORDER_CONSTANT, value=0),
                    albu.HorizontalFlip(always_apply=False, p=prob),
                    albu.VerticalFlip(always_apply=False, p=prob),
                    albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=(1, 1), rotate_limit=(0, 0), interpolation=1,
                                          border_mode=4, value=None, mask_value=None, always_apply=False, p=prob)],
                   p=prob),
        albu.OneOf([albu.GridDistortion(p=prob, distort_limit=(-0.2, 0), num_steps=5),
                    albu.ElasticTransform(p=prob, alpha_affine=30),
                    albu.Perspective(scale=(0.01, 0.05), p=prob)], p=prob),
        albu.OneOf([albu.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5,
                                                  brightness_by_max=None, always_apply=False, p=prob),
                    albu.GaussNoise(p=prob, var_limit=(20, 100)),
                    albu.MotionBlur(blur_limit=15, always_apply=False, p=prob),
                    albu.GaussianBlur(blur_limit=15, always_apply=False, p=prob),
                    albu.RandomGamma(gamma_limit=(40, 160), eps=1e-07, always_apply=False, p=prob),
                    albu.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=prob),
                    albu.RGBShift(always_apply=False, p=prob, r_shift_limit=(-20, 20), g_shift_limit=(-20, 20),
                                  b_shift_limit=(-20, 20))], p=prob)]
    return albu.Compose(train_transform)


class ClassificationDataset(Dataset):
    def __init__(self,
                 img_path,
                 category_num,
                 train=False,
                 test=False,
                 transforms=None):

        self.img_path = img_path
        self.train = train
        self.transforms = transforms
        self.albu_transforms = augment_compose(0.5)
        self.test = test

        self.img_name = []
        self.labels = []
        if self.train:
            for img_fold in self.img_path:
                for cla in os.listdir(img_fold):
                    for p in os.listdir(os.path.join(img_fold, cla)):
                        for img_name in os.listdir(os.path.join(img_fold, cla, p)):
                            self.img_name.append(os.path.join(img_fold, cla, p, img_name))
                            self.labels.append(int(cla))
        self.length = len(self.labels)

    def __getitem__(self, index):
        # print(self.labels[index])
        img = Image.open(self.img_name[index]).convert('RGB')
        if self.transforms is not None:
            try:
                img = self.transforms(img)
            except:
                print("Cannot transform image: {}".format(
                    self.img_name[index]))
        return (img, self.labels[index], self.img_name[index])

        # if self.albu_transforms is not None:
        #     try:
        #         img = self.albu_transforms(image=np.array(img))
        #     except Exception as e:
        #         print("Cannot transform image: {}".format(self.img_name[index]), ': for error in ', e)
        # return image_transforms['valid'](img['image']), self.labels[index], self.img_name[index]

    def __len__(self):
        return self.length


class FusionDataset(Dataset):
    def __init__(self,
                 img_path,
                 category_num,
                 train=False,
                 test=False,
                 transforms=None):

        self.img_path = img_path
        self.train = train
        self.transforms = transforms
        self.albu_transforms = augment_compose(0.5)
        self.test = test

        # self.img_name = []
        self.gray_img, self.blood_img = [], []
        self.labels = []
        if self.train:
            for img_fold in self.img_path:
                for cla in os.listdir(img_fold):
                    for p in os.listdir(os.path.join(img_fold, cla)):
                        img_name = os.listdir(os.path.join(img_fold, cla, p))
                        self.gray_img.append(os.path.join(img_fold, cla, p, img_name[0]))
                        self.blood_img.append(os.path.join(img_fold, cla, p, img_name[1]))
                        self.labels.append(int(cla))
        self.length = len(self.labels)

    def __getitem__(self, index):
        img1 = Image.open(self.gray_img[index]).convert('RGB')
        img2 = Image.open(self.blood_img[index]).convert('RGB')
        if self.transforms is not None:
            try:
                img1 = self.transforms(img1)
                img2 = self.transforms(img2)
            except:
                print("Cannot transform image: {}".format(
                    self.gray_img[index]))
        return img1, img2, self.labels[index], self.gray_img[index]

    def __len__(self):
        return self.length


class EarlyAddFusionModel(nn.Module):
    def __init__(self, net):
        super(EarlyAddFusionModel, self).__init__()
        self.net = net

    def forward(self, x1, x2):
        x = torch.add(x1, x2)
        x = self.net(x)
        return x


class EarlyCatFusionModel(nn.Module):
    def __init__(self, net):
        super(EarlyCatFusionModel, self).__init__()
        self.net = net
        self.net.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.net(x)
        return x


class LateAddFusionModel(nn.Module):
    def __init__(self, net, num_class):
        super(LateAddFusionModel, self).__init__()
        self.net = net
        self.fc = nn.Linear(in_features=2048, out_features=num_class, bias=True)

    def forward(self, x1, x2):
        x1 = self.net.conv1(x1)
        x1 = self.net.bn1(x1)
        x1 = self.net.relu(x1)
        x1 = self.net.maxpool(x1)
        x1 = self.net.layer1(x1)
        x1 = self.net.layer2(x1)
        x1 = self.net.layer3(x1)
        x1 = self.net.layer4(x1)

        x2 = self.net.conv1(x2)
        x2 = self.net.bn1(x2)
        x2 = self.net.relu(x2)
        x2 = self.net.maxpool(x2)
        x2 = self.net.layer1(x2)
        x2 = self.net.layer2(x2)
        x2 = self.net.layer3(x2)
        x2 = self.net.layer4(x2)

        x = torch.add(x1, x2)
        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class LateCatFusionModel(nn.Module):
    def __init__(self, net, num_class):
        super(LateCatFusionModel, self).__init__()
        self.net = net
        self.fc = nn.Linear(in_features=4096, out_features=num_class, bias=True)

    def forward(self, x1, x2):
        x1 = self.net.conv1(x1)
        x1 = self.net.bn1(x1)
        x1 = self.net.relu(x1)
        x1 = self.net.maxpool(x1)
        x1 = self.net.layer1(x1)
        x1 = self.net.layer2(x1)
        x1 = self.net.layer3(x1)
        x1 = self.net.layer4(x1)

        x2 = self.net.conv1(x2)
        x2 = self.net.bn1(x2)
        x2 = self.net.relu(x2)
        x2 = self.net.maxpool(x2)
        x2 = self.net.layer1(x2)
        x2 = self.net.layer2(x2)
        x2 = self.net.layer3(x2)
        x2 = self.net.layer4(x2)

        x = torch.cat((x1, x2), dim=1)
        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class AttentionFusionModel(nn.Module):
    def __init__(self, net, num_class, modality1_channels, modality2_channels):
        super(AttentionFusionModel, self).__init__()

        # 提取 ResNet50 的前半部分作为特征提取器
        self.features = nn.Sequential(*list(net.children())[:-2])

        # 修改第一个卷积层，使其适应多模态输入
        self.features[0] = nn.Conv2d(modality1_channels + modality2_channels, 64, kernel_size=(7, 7), stride=(2, 2),
                                     padding=(3, 3), bias=False)

        # 全局平均池化层
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(2048, 64),  # 64是ResNet50的输出通道数，4是因为有两个模态
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )

        # 分类层
        self.classifier = nn.Linear(2048, num_class, bias=True)

    def forward(self, x1, x2):
        # 拼接两个模态的输入
        x = torch.cat((x1, x2), dim=1)

        # 提取特征
        features = self.features(x)

        # 全局平均池化
        pooled_features = self.global_avg_pooling(features)

        # 展开特征
        pooled_features = pooled_features.view(pooled_features.size(0), -1)

        # 注意力权重
        attention_weights = self.attention(pooled_features)

        # 加权融合特征
        fused_features = torch.sum(features * attention_weights.view(-1, 1, 1, 1), dim=(2, 3))

        # 分类
        output = self.classifier(fused_features)

        return output


def prepare_model(category_num, model_name, lr, num_epochs, device, weights):
    if 'eca' in model_name:  # ECA（Efficient Channel Attention）
        model = model_dict[model_name]()
    elif 'CBAM_Resnext' in model_name:
        if model_name == 'CBAM_Resnext50':
            model = model_dict[model_name](50, category_num)
        elif model_name == 'CBAM_Resnext101':
            model = model_dict[model_name](101, category_num)
        elif model_name == 'CBAM_Resnext152':
            model = model_dict[model_name](152, category_num)
    elif 'SENet' in model_name:
        model = model_dict[model_name](pretrained='imagenet')
    else:
        model = model_dict[model_name](pretrained=True)

    if model_name in ['resnet50', 'resnet101', 'resnet152', 'resnext50', 'wide_resnet50', 'resnext101',
                      'wide_resnet101', 'eca_resnet50', 'eca_resnext50', 'resnet50_cbam', 'resnet101_cbam',
                      'resnet152_cbam']:
        model.fc = nn.Linear(in_features=2048, out_features=category_num, bias=True)
    elif model_name == ['resnet18', 'resnet18_cbam']:
        model.fc = nn.Linear(in_features=512, out_features=category_num, bias=True)
    elif model_name == 'densenet121':
        model.classifier = nn.Linear(in_features=1024, out_features=category_num, bias=True)
    elif model_name == 'densenet161':
        model.classifier = nn.Linear(in_features=2208, out_features=category_num, bias=True)
    elif model_name == 'densenet169':
        model.classifier = nn.Linear(in_features=1664, out_features=category_num, bias=True)
    elif model_name == 'densenet201':
        model.classifier = nn.Linear(in_features=1920, out_features=category_num, bias=True)
    elif model_name == 'regnet_y_32gf':
        model.fc = nn.Linear(in_features=3712, out_features=category_num, bias=True)
    elif model_name in ['vgg16', 'vgg11']:
        model.classifier[6] = nn.Linear(in_features=4096, out_features=category_num, bias=True)
    elif model_name in ['efficientnet_b0']:
        model.classifier[1] = nn.Linear(in_features=1280, out_features=category_num, bias=True)
    elif model_name in ['SENet_resnext50', 'SENet_resnet50']:  # SENet
        model.last_linear = nn.Linear(in_features=2048, out_features=category_num, bias=True)
    elif model_name in ['CBAM_resnext50', 'CBAM_resnext101', 'CBAM_resnext152']:  # ECA
        model.layer7 = nn.Linear(in_features=2048, out_features=category_num, bias=True)

    'fusion'
    # model = EarlyCatFusionModel(model)
    # model = LateCatFusionModel(model, category_num)
    # model = AttentionFusionModel(model, category_num, 3, 3)

    # 多GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    # 定义损失函数和优化器。
    # 定义loss权重  class_weights = torch.tensor([5.0, 1.0])
    loss_func = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=2e-4, momentum=0.9, nesterov=True)
    # optimizer = optim.NAdam(model.parameters(), lr=lr, betas=(0.8, 0.888), eps=1e-08, weight_decay=2e-4)
    # 定义学习率与轮数关系的函数
    # lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.005)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=lr_scheduler)
    return model, optimizer, lr_scheduler, scheduler_warmup, loss_func


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, 'best_network.pth')
        torch.save(model.state_dict(), path)  # 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss
