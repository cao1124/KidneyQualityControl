import collections
import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from data_preprocess import cv_read, cv_write
from segment_util import KidneyMassDataset, training_augmentation, valid_augmentation, save_seg_history, \
    add_weighted_multi, combine_image, get_iou, get_f1


def train(data_dir, encoder_name, encoder_activation, bs, lr, epochs, save_dir, device):
    for i in range(5):
        save_dir1 = save_dir + "fold" + str(i) + '/'
        os.makedirs(save_dir1, exist_ok=True)
        print('五折交叉验证 第{}次实验:'.format(i))
        fold_list = ['fold0/', 'fold1/', 'fold2/', 'fold3/', 'fold4/',
                     'fold5/', 'fold6/', 'fold7/', 'fold8/', 'fold9/']
        valid_path = [data_dir + fold_list[i],
                      data_dir + fold_list[i+1]]
        valid_mask = [data_dir.replace('kfold', 'mask-npy') + fold_list[i],
                      data_dir.replace('kfold', 'mask-npy') + fold_list[i+1]]
        fold_list.remove(fold_list[i])
        fold_list.remove(fold_list[i+1])
        test_path = [data_dir + fold_list[i]]
        test_mask = [data_dir.replace('kfold', 'mask-npy') + fold_list[i]]
        fold_list.remove(fold_list[i])
        train_path, train_mask = [], []
        for x in range(len(fold_list)):
            train_path.append(data_dir + fold_list[x])
            train_mask.append(data_dir.replace('kfold', 'mask-npy') + fold_list[x])

        train_dataset = KidneyMassDataset(train_path, train_mask, augmentation=training_augmentation())
        valid_dataset = KidneyMassDataset(valid_path, valid_mask, augmentation=valid_augmentation())
        test_dataset = KidneyMassDataset(test_path, test_mask, augmentation=valid_augmentation())
        print('train size:{}, valid:{}, test:{}'.format(len(train_dataset), len(valid_dataset), len(test_dataset)))
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=4)

        # build model
        model = smp.DeepLabV3Plus(encoder_name=encoder_name,
                         classes=2,
                         activation=encoder_activation,
                         in_channels=3,
                         encoder_weights="imagenet")
        # print(model)
        loss_fn = smp.utils.losses.DiceLoss() + smp.utils.losses.BCELoss()
        # for image segmentation dice loss could be the best first choice
        # loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        metrics = [
            smp.utils.metrics.IoU(threshold=0.5),
            smp.utils.metrics.Fscore()
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
        early_stops = 2000

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
        # colors = [("renal", [255, 0, 255]), ("mass", [128, 0, 255]), ("reference", [255, 0, 128])]
        # colors_mask = [("renal", [64, 64, 64]), ("mass", [128, 128, 128]), ("reference", [160, 160, 160])]
        iou_list, dice_list = [], []
        renal_iou_list, renal_dice_list = [], []
        mass_iou_list, mass_dice_list = [], []
        print('load model name:', [x for x in os.listdir(save_dir1) if x.endswith('.pth')][-1])
        model = torch.load(save_dir1 + [x for x in os.listdir(save_dir1) if x.endswith('.pth')][-1])
        # model = torch.load('C:/Users/user/Desktop/0731-segment-efficientnet-b7/fold0/best_0.6564.pth')
        model.eval()
        torch.cuda.empty_cache()  # 释放缓存分配器当前持有的且未占用的缓存显存
        with torch.no_grad():
            for k in range(len(test_dataset)):
                image, gt_mask2 = test_dataset[k]
                gt_mask2 = gt_mask2.squeeze().astype(np.uint8)
                mask_ori = cv_read(os.path.join(test_dataset.masks[k]))
                [orig_h, orig_w, orig_c] = mask_ori.shape
                x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
                pred_mask2 = model(x_tensor)
                pred_mask2 = (pred_mask2.squeeze().cpu().numpy().round().astype(np.uint8))
                pred_mask2[pred_mask2 < 0.5] = 0
                pred_mask2[pred_mask2 >= 0.5] = 1
                pred_draw = np.zeros([orig_h, orig_w, 3]).astype(np.uint8)
                gt_draw = np.zeros([orig_h, orig_w, 3]).astype(np.uint8)
                for c in range(orig_c):
                    gt_mask = gt_mask2[c]
                    pred_mask = pred_mask2[c]

                    if np.sum(mask_ori) == 0 and np.sum(pred_mask) == 0:
                        iou = 1
                        dice = 1
                    else:
                        iou = get_iou(gt_mask, pred_mask)
                        dice = get_f1(gt_mask, pred_mask)

                    if c == 0:
                        name = 'renal'
                        renal_iou_list.append(np.round(iou, 4))
                        renal_dice_list.append(np.round(dice, 4))
                    elif c == 1:
                        name = 'mass'
                        mass_iou_list.append(np.round(iou, 4))
                        mass_dice_list.append(np.round(dice, 4))
                    print(test_dataset.images[k], "\t", name, ":dice:", dice, "\tiou:", iou)

                    if pred_mask.shape != mask_ori.shape:
                        pred_mask = cv2.resize(pred_mask, (orig_w, orig_h), cv2.INTER_NEAREST)
                        gt_mask = cv2.resize(gt_mask, (orig_w, orig_h), cv2.INTER_NEAREST)
                        pred_draw[:, :, c] = pred_mask
                        gt_draw[:, :, c] = gt_mask
                save_full_path = save_dir1 + os.path.split(test_dataset.images[k])[0].split('/')[-1] + '/' + \
                                 os.path.split(test_dataset.images[k])[1]
                print(save_full_path)
                os.makedirs(os.path.split(save_full_path)[0], exist_ok=True)
                # cv2.imwrite(save_full_path, pred_mask)

                img = cv_read(test_dataset.images[k], cv2.IMREAD_COLOR)
                img_gt = add_weighted_multi(img, gt_draw, 'BGR')
                img_pred = add_weighted_multi(img, pred_draw, 'BGR')
                img_gt_pred = combine_image(img_gt, img_pred)
                cv_write(save_full_path, img_gt_pred)

        print("\tRenal Mean Dice:", np.average(renal_dice_list))
        print("\tRenal Mean IoU:", np.average(renal_iou_list))
        print("\tMass Mean Dice:", np.average(mass_dice_list))
        print("\tMass Mean IoU:", np.average(mass_iou_list))
        # hist, bins = np.histogram(dice_list, bins=np.arange(0.0, 1.05, 0.1))
        # print(hist)
        # print(hist / len(test_dataset))


def segment():
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = '/home/ai999/dataset/kidney/kidney-mass-kfold/'
    # 'D:/med dataset/kidney-small-tumor-kfold/'     # '/home/ai999/dataset/kidney/kidney-small-tumor-kfold/'
    encoder_name = "efficientnet-b7"               # "efficientnet-b7"  'resnext50_32x4d'
    encoder_activation = "softmax2d"  # could be None for logits or 'softmax2d' for multiclass segmentation
    # encoder_weights = "imagenet"
    # preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name, encoder_weights)
    bs = 5
    lr = 1e-4
    epochs = 10000
    save_dir = "kidney-mass-segment/0804-deeplabv3-segment-" + encoder_name + '/'
    train(data_dir, encoder_name, encoder_activation, bs, lr, epochs, save_dir, device)


if __name__ == '__main__':
    segment()
