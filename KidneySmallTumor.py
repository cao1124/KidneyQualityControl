import collections
import copy
import os
from enum import Enum

import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from data_preprocess import cv_read, cv_write
from multi_class_loss import MulticlassDiceLoss
from segment_util import RenalMassDataset, training_augmentation, valid_augmentation, save_seg_history, \
    add_weighted_multi, combine_image, get_iou, get_f1, add_weighted


def train(data_dir, encoder_name, encoder_activation, bs, lr, epochs, save_dir, device, target_list):
    for i in range(5):
        save_dir1 = os.path.join(save_dir, "fold" + str(i))
        os.makedirs(save_dir1, exist_ok=True)
        print('五折交叉验证 第{}次实验:'.format(i))
        test_path = [os.path.join(data_dir, 'fold4/')]
        fold_list = ['fold0/', 'fold1/', 'fold2/', 'fold3/']
        valid_path = [os.path.join(data_dir, fold_list[3 - i])]
        train_path = [os.path.join(data_dir, x) for x in fold_list if x != valid_path[0]]
        '随机test database'
        # fold_list = ['fold0/', 'fold1/', 'fold2/', 'fold3/', 'fold4/']
        # valid_path = [os.path.join(data_dir, fold_list[i])]
        # fold_list.remove(fold_list[i])
        # if i == 4:
        #     test_path = [os.path.join(data_dir, fold_list[0])]
        #     fold_list.remove(fold_list[0])
        # else:
        #     test_path = [os.path.join(data_dir, fold_list[i])]
        #     fold_list.remove(fold_list[i])
        # train_path = []
        # for x in range(len(fold_list)):
        #     train_path.append(os.path.join(data_dir, fold_list[x]))

        train_dataset = RenalMassDataset(data_dir, train_path, len(target_list) + 1, augmentation=training_augmentation())
        valid_dataset = RenalMassDataset(data_dir, valid_path, len(target_list) + 1, augmentation=valid_augmentation())
        test_dataset = RenalMassDataset(data_dir, test_path, len(target_list) + 1, augmentation=valid_augmentation())
        print('train size:{}, valid:{}, test:{}'.format(len(train_dataset), len(valid_dataset), len(test_dataset)))
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=4)

        # build model
        model = smp.Unet(encoder_name=encoder_name,
                         classes=len(target_list) + 1,
                         activation=encoder_activation,
                         in_channels=3,
                         encoder_weights="imagenet")
        loss_fn = MulticlassDiceLoss()

        metrics = [
            smp.utils.metrics.IoU(threshold=0.5, ignore_channels=[0]),
            smp.utils.metrics.Fscore(ignore_channels=[0])
        ]

        optimizer = torch.optim.Adam([
            dict(params=model.parameters(), lr=lr),
        ])

        # create epoch runners
        # it is a simple loop of iterating over dataloader`s samples
        train_epoch = smp.utils.train.TrainEpoch(
            model,
            loss=loss_fn,
            metrics=metrics,
            optimizer=optimizer,
            device=device,
            verbose=True,
        )

        valid_epoch = smp.utils.train.ValidEpoch(
            model,
            loss=loss_fn,
            metrics=metrics,
            device=device,
            verbose=True,
        )

        # train model for 40 epochs
        max_score = -1
        max_dice = 0
        best_epoch = 0
        early_stops = 500

        train_history = {'dice_loss + bce_loss': [], 'fscore': []}
        val_history = {'dice_loss + bce_loss': [], 'fscore': []}
        for j in range(epochs):
            if j - best_epoch > early_stops:
                print(j - best_epoch, " epochs don't change, early stopping.")
                break
            print('\nEpoch: {}'.format(j))
            print("Best epoch:", best_epoch, "\tiou:", max_score, "\tbest dice:", max_dice)
            train_logs = train_epoch.run(train_loader)
            train_history['dice_loss + bce_loss'].append(train_logs['dice_loss + bce_loss'])
            train_history['fscore'].append(train_logs['fscore'])

            valid_logs = valid_epoch.run(valid_loader)
            val_history['dice_loss + bce_loss'].append(valid_logs['dice_loss + bce_loss'])
            val_history['fscore'].append(valid_logs['fscore'])

            save_seg_history(train_history, val_history, save_dir1)

            # do something (save model, change lr, etc.)
            if max_score < np.round(valid_logs['iou_score'], 4):  # fscore  iou_score
                if max_score != -1:
                    old_filepath = save_dir1 + "best_" + str(max_score) + ".pth"
                    os.remove(old_filepath)
                max_score = np.round(valid_logs['iou_score'], 4)
                max_dice = np.round(valid_logs['fscore'], 4)
                torch.save(model, save_dir1 + "best_" + str(max_score) + ".pth")
                print('best iou score={}, Model saved!'.format(max_score))
                best_epoch = j

            if j - best_epoch > 1000:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 2
                print('Decrease decoder learning rate. lr:', optimizer.param_groups[0]['lr'])

        'test'
        color_list = ['BG', 'GR']
        iou_list, dice_list = [], []
        renal_iou_list, renal_dice_list = [], []
        mass_iou_list, mass_dice_list = [], []
        model.eval()
        torch.cuda.empty_cache()  # 释放缓存分配器当前持有的且未占用的缓存显存

        for k in range(len(test_dataset)):
            img_iou, img_dice = [], []
            image, gt_mask2 = test_dataset[k]
            gt_mask = cv_read(os.path.splitext(test_dataset.images[k].replace('classify', 'segment'))[0] + '.jpg')
            image_ori = cv_read(test_dataset.images[k], 1)
            [orig_h, orig_w, _] = image_ori.shape

            x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
            img_pred = copy.deepcopy(image_ori)
            img_gt = copy.deepcopy(image_ori)
            with torch.no_grad():
                pr_mask = model(x_tensor).squeeze(0).cpu().numpy()
            for c in range(1, len(target_list) + 1):
                mask_gt = copy.deepcopy(gt_mask)
                if c == 1:
                    mask_gt[mask_gt != 128] = 0
                    mask_gt[mask_gt == 128] = 255
                else:
                    mask_gt[mask_gt != 255] = 0
                    mask_gt[mask_gt == 255] = 255

                pred_mask = pr_mask[c]
                pred_mask[pred_mask < 0.5] = 0
                pred_mask[pred_mask >= 0.5] = 255
                if pred_mask.shape != image_ori.shape:
                    pred_mask = cv2.resize(pred_mask, (orig_w, orig_h), cv2.INTER_NEAREST)
                    _, pred_mask = cv2.threshold(pred_mask, 1, 255, cv2.THRESH_BINARY)
                img_pred = add_weighted(img_pred, pred_mask.astype('uint8'), color_list[c - 1])

                # if np.sum(mask_gt) == 0 and np.sum(pred_mask) == 0:
                #     iou = 1
                #     dice = 1
                # else:
                #     iou = np.round(get_iou(mask_gt, pred_mask), 4)
                #     dice = np.round(get_f1(mask_gt, pred_mask), 4)
                if np.sum(mask_gt) != 0 or np.sum(pred_mask) != 0:
                    iou = np.round(get_iou(mask_gt, pred_mask), 4)
                    dice = np.round(get_f1(mask_gt, pred_mask), 4)
                    iou_list.append(iou)
                    dice_list.append(dice)
                    img_iou.append(iou)
                    img_dice.append(dice)
                    if c == 1:
                        renal_iou_list.append(iou)
                        renal_dice_list.append(dice)
                    elif c == 2:
                        mass_iou_list.append(iou)
                        mass_dice_list.append(dice)
                img_gt = add_weighted(img_gt, mask_gt.astype('uint8'), color_list[c - 1])
            save_full_path = os.path.join(save_dir1, str(np.average(img_iou)) + "-" + test_dataset.images[k].split('/')[-1])
            img_cat = np.concatenate((img_gt, img_pred), axis=1)
            cv_write(save_full_path, img_cat)

        print("\tRenal Mean Dice: ", np.average(renal_dice_list), "Renal Mean IoU:", np.average(renal_iou_list))
        print("\tMass Mean Dice: ", np.average(mass_dice_list), "Mass Mean IoU:", np.average(mass_iou_list))
        print("\tAll Mean Dice: ", np.average(mass_dice_list), "All Mean IoU:", np.average(mass_iou_list))


def segment():
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = '/media/user/Disk1/caoxu/dataset/kidney/20231204-segment-dataset/image-5fold/'
    encoder_name = "efficientnet-b0"
    encoder_activation = "softmax2d"
    target_list = [x.name for x in RenalMass]
    bs = 6
    lr = 1e-4
    epochs = 2000
    save_dir = "kidney-mass-segment/20231205-unet-segment-" + encoder_name + '/'
    train(data_dir, encoder_name, encoder_activation, bs, lr, epochs, save_dir, device, target_list)


class RenalMass(Enum):
    renal = 1
    mass = 2


if __name__ == '__main__':
    segment()
