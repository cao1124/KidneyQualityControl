import os
from PIL import Image
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset
from torchvision import transforms, models
from warmup_scheduler import GradualWarmupScheduler

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
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mass_mean, mass_std)
    ])
}


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
        self.length = len(self.img_name)

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

    def __len__(self):
        return self.length


def prepare_model(category_num, model_name, lr, num_epochs, device):
    model = model_dict[model_name](pretrained=True)
    if model_name in ['resnet50', 'resnet101', 'resnet152', 'resnext50', 'wide_resnet50', 'resnext101',
                      'wide_resnet101']:
        model.fc = nn.Linear(in_features=2048, out_features=category_num, bias=True)
    elif model_name == 'resnet18':
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

    # 多GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    # 定义损失函数和优化器。
    loss_func = nn.CrossEntropyLoss()
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
        torch.save(model.state_dict(), path)	# 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss

