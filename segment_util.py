import collections

import torch
from torch.utils.data import Dataset as BaseDataset
import cv2
import os
import albumentations as albu
from sklearn.metrics import jaccard_score, f1_score
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from data_preprocess import cv_read

matplotlib.use('Agg')


def save_seg_history(train_logs, val_logs, save_file_name):
    plt.plot(train_logs['dice_loss + bce_loss'], '--')
    plt.plot(train_logs['fscore'], '--')
    plt.plot(val_logs['dice_loss + bce_loss'])
    plt.plot(val_logs['fscore'])

    plt.title("Model loss & DICE")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['train loss', 'train DICE', 'val loss', 'val DICE'], loc='upper right')
    plt.savefig(save_file_name + "_model_history.png")
    plt.close()


def training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        # albu.PadIfNeeded(p=1,min_height=496, min_width=640, always_apply=True, border_mode=0),
        # albu.RandomCrop(height=480, width=480, always_apply=True),
        # albu.IAAAdditiveGaussianNoise(p=0.2),
        # albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.5,
        ),

        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.5,
        ),

        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=1),
                # albu.HueSaturationValue(p=1),
            ],
            p=0.5,
        ),
    ]
    return albu.Compose(train_transform)


def valid_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        #  albu.PadIfNeeded(min_height=496, min_width=640),
        #  albu.RandomCrop(height=480, width=480, always_apply=True)
    ]
    return albu.Compose(test_transform)


def get_f1(gt, pred):
    gt = gt.ravel()
    pred = pred.ravel()

    f1 = f1_score(gt, pred, average=None)

    if len(f1) == 1:
        return f1[0]
    return f1[1]


def get_iou(gt, pred):
    gt = gt.ravel()
    pred = pred.ravel()
    iou = jaccard_score(gt, pred, average=None)

    if len(iou) == 1:
        return iou[0]
    return iou[1]


def add_weighted(img, mask, which_channel="B", w_alpha=0.2):
    [B, G, R] = cv2.split(img)
    if "B" in which_channel:
        B = cv2.addWeighted(B, 1, mask, w_alpha, 0)
    if "G" in which_channel:
        G = cv2.addWeighted(G, 1, mask, w_alpha, 0)
    if "R" in which_channel:
        R = cv2.addWeighted(R, 1, mask, w_alpha, 0)
    img_add = cv2.merge([B, G, R])
    return img_add


def add_weighted_multi(img, mask, which_channel="B", w_alpha=0.2):
    [B, G, R] = cv2.split(img)
    [B_m, G_m, R_m] = cv2.split(mask)
    if "B" in which_channel:
        B_m[B_m == 1] = 255
        B = cv2.addWeighted(B, 1, B_m, w_alpha, 0)
    if "G" in which_channel:
        G_m[G_m == 1] = 255
        G = cv2.addWeighted(G, 1, G_m, w_alpha, 0)
    if "R" in which_channel:
        R_m[R_m == 1] = 255
        R = cv2.addWeighted(R, 1, R_m, w_alpha, 0)
    img_add = cv2.merge([B, G, R])
    return img_add


def combine_image(gt, pred):
    [height, width, channel] = gt.shape
    img_res = np.zeros((height, width * 2, channel), dtype=np.int32)
    img_res[:height, :width, :] = gt
    img_res[:height, width:width * 2, :] = pred
    return img_res.astype('uint8')


class RenalMassDataset(BaseDataset):
    def __init__(
            self,
            base_dir,
            images_dir,
            num_class,
            augmentation=None,
            preprocessing=None,
            multi_scale=False,
    ):
        self.images, self.masks = [], []
        for f in images_dir:
            for c in os.listdir(os.path.join(base_dir, f)):
                for p in os.listdir(os.path.join(base_dir, f, c)):
                    for img_name in [x for x in os.listdir(os.path.join(base_dir, f, c, p)) if not x.endswith('.json')]:
                        self.images.append(os.path.join(f, c, p, img_name))
                        self.masks.append(os.path.join(f.replace('classify', 'segment'), c, p, os.path.splitext(img_name)[0] + '.png'))
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.multi_scale = multi_scale
        self.num_classes = num_class

    def __getitem__(self, item):
        if type(item) == list or type(item) == tuple:
            i, size = item
        else:
            i = item
        image = cv_read(self.images[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv_read(self.masks[i])

        target_width = 512
        target_height = 512
        if image.shape[:2] != [target_height, target_width]:
            image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
            # _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        image = image / 255.0

        mask = mask[np.newaxis, :, :]
        mask[mask == 128] = 1
        mask[mask == 255] = 2
        label_true = torch.LongTensor(mask)
        try:
            one_hot = torch.zeros(self.num_classes, mask.shape[1], mask.shape[2]).scatter_(0, label_true, 1)
        except Exception as e:
            print(self.images[i], e)
        image = image.transpose(2, 0, 1).astype('float32')
        return image, one_hot

    def __len__(self):
        return len(self.images)


class RenalDataset(BaseDataset):
    def __init__(
            self,
            images_dir,
            masks_dir,
            augmentation=None,
            preprocessing=None,
            multi_scale=False,
    ):
        self.images, self.masks = [], []
        for i in range(len(images_dir)):
            for cla in os.listdir(images_dir[i]):
                # for p in os.listdir(os.path.join(images_dir[i], cla)):
                for img_name in os.listdir(os.path.join(images_dir[i], cla)):
                    if os.path.exists(os.path.join(masks_dir[i], cla, img_name)):
                        self.images.append(os.path.join(images_dir[i], cla, img_name))
                        self.masks.append(os.path.join(masks_dir[i], cla, img_name))
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.multi_scale = multi_scale

    def __getitem__(self, item):
        if type(item) == list or type(item) == tuple:
            i, size = item
        else:
            i = item
        # read data
        image = cv_read(self.images[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv_read(self.masks[i])

        target_width = 512
        target_height = 512
        if image.shape[:2] != [target_height, target_width]:
            image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        image = image / 255.0
        mask[mask == 255] = 1

        image = image.transpose(2, 0, 1).astype('float32')
        mask = np.expand_dims(mask, 0).astype('float32')

        return image, mask

    def __len__(self):
        return len(self.images)
