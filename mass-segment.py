import os
import cv2
import torch
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from segment_util import SegmentDataset, training_augmentation, valid_augmentation, save_seg_history, get_iou, get_f1, \
    add_weighted, combine_image


def train(data_dir, encoder_name, encoder_activation, bs, lr, epochs, save_dir, device):
    for i in range(5):
        save_dir1 = save_dir + "fold" + str(i) + '/'
        os.makedirs(save_dir1, exist_ok=True)
        print('五折交叉验证 第{}次实验:'.format(i))
        fold_list = ['fold0/', 'fold1/', 'fold2/', 'fold3/', 'fold4/']
        mask_dir = data_dir.replace('kidney-mass-kfold/', 'mass-mask/')
        valid_path = [data_dir + fold_list[i]]
        valid_mask = [mask_dir + fold_list[i]]
        fold_list.remove(fold_list[i])
        if i == 4:
            test_path = [data_dir + fold_list[0]]
            test_mask = [mask_dir + fold_list[0]]
            fold_list.remove(fold_list[0])
        else:
            test_path = [data_dir + fold_list[i]]
            test_mask = [mask_dir + fold_list[i]]
            fold_list.remove(fold_list[i])
        train_path = [data_dir + fold_list[0], data_dir + fold_list[1], data_dir + fold_list[2]]
        train_mask = [mask_dir + fold_list[0], mask_dir + fold_list[1], mask_dir + fold_list[2]]

        train_dataset = SegmentDataset(train_path, train_mask, augmentation=training_augmentation(), muilt_scale=False)
        valid_dataset = SegmentDataset(valid_path, valid_mask, augmentation=valid_augmentation(), muilt_scale=False)
        test_dataset = SegmentDataset(test_path, test_mask, augmentation=valid_augmentation(), muilt_scale=False)
        print('train size:{}, valid:{}, test:{}'.format(len(train_dataset), len(valid_dataset), len(test_dataset)))
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=4)

        # build model
        model = smp.Unet(encoder_name=encoder_name,
                         classes=1,
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
        iou_list, dice_list = [], []
        print('model_name:', [x for x in os.listdir(save_dir1) if x.endswith('.pth')][-1])
        model = torch.load(save_dir1 + [x for x in os.listdir(save_dir1) if x.endswith('.pth')][-1])
        model.eval()
        torch.cuda.empty_cache()  # 释放缓存分配器当前持有的且未占用的缓存显存
        for k in range(len(test_dataset)):
            image, gt_mask = test_dataset[k]
            gt_mask = gt_mask.squeeze()
            gt_mask[gt_mask == 1] = 255
            x_tensor = torch.from_numpy(image).to(device).unsqueeze(0)

            mask_ori = cv2.imread(os.path.join(test_dataset.masks[k]), cv2.IMREAD_GRAYSCALE)
            _, mask_ori = cv2.threshold(mask_ori, 1, 255, cv2.THRESH_BINARY)
            [orig_h, orig_w] = mask_ori.shape

            with torch.no_grad():
                pred_mask = model(x_tensor)
                pred_mask = (pred_mask.squeeze().cpu().numpy().round())

            pred_mask[pred_mask < 0.5] = 0
            pred_mask[pred_mask >= 0.5] = 255

            if pred_mask.shape != mask_ori.shape:
                pred_mask = cv2.resize(pred_mask, (orig_w, orig_h), cv2.INTER_NEAREST)
                gt_mask = cv2.resize(gt_mask, (orig_w, orig_h), cv2.INTER_NEAREST)
                _, pred_mask = cv2.threshold(pred_mask, 1, 255, cv2.THRESH_BINARY)

            if np.sum(mask_ori) == 0 and np.sum(pred_mask) == 0:
                iou = 1
                dice = 1
            else:
                iou = get_iou(mask_ori, pred_mask)
                dice = get_f1(mask_ori, pred_mask)

            iou = np.round(iou, 4)
            dice = np.round(dice, 4)
            iou_list.append(iou)
            dice_list.append(dice)
            print(test_dataset.images[k], "\tdice:", dice, "\tiou:", iou)

            save_full_path = save_dir1 + test_dataset.images[k].split('/')[-1]    # windows \\   linux /
            print(save_full_path)
            # cv2.imwrite(save_full_path, pred_mask)

            img = cv2.imread(test_dataset.images[k], cv2.IMREAD_COLOR)
            img_gt = add_weighted(img, gt_mask.astype('uint8'), 'BG')
            img_pred = add_weighted(img, pred_mask.astype('uint8'), 'GR')
            img_gt_pred = combine_image(img_gt, img_pred)
            cv2.imwrite(save_full_path, img_gt_pred)

        print("\tMean Dice:", np.average(dice_list))
        print("\tMean IoU:", np.average(iou_list))
        # hist, bins = np.histogram(dice_list, bins=np.arange(0.0, 1.05, 0.1))
        # print(hist)
        # print(hist / len(test_dataset))


def segment():
    os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = '/home/ai999/dataset/kidney/kidney-mass-kfold/'
    """
    分割网络选择：
    Unet、Linknet、FPN、PSPNet、PAN、DeepLabV3、UnetPlusPlus
    backbone选择：
    resnet18  34   50  101   152
    resnext50_32x4d resnext101_32x8d resnext101_32x16d resnext101_32x32d resnext101_32x48d
    dpn68 dpn68b dpn92 dpn98 dpn107 dpn131
    vgg11 vgg11_bn vgg13 vgg13_bn vgg16 vgg16_bn vgg19 vgg19_bn
    senet154
    se_resnet50 se_resnet101 se_resnet152
    se_resnext50_32x4d se_resnext101_32x4d
    densenet121 densenet169 densenet201 densenet161
    inceptionresnetv2 inceptionv4
    efficientnet-b0 b7
    mobilenet_v2
    xception
    timm-efficientnet-b0 b8
    timm-efficientnet-l2
    replknet-31b
    """
    encoder_name = "efficientnet-b7"
    encoder_weights = "imagenet"
    encoder_activation = "sigmoid"  # could be None for logits or 'softmax2d' for multiclass segmentation
    # preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name, encoder_weights)
    bs = 24
    lr = 1e-4
    epochs = 10000
    save_dir = "mass-segment/0804-deeplabv3-segment-" + encoder_name + '/'
    train(data_dir, encoder_name, encoder_activation, bs, lr, epochs, save_dir, device)


if __name__ == '__main__':
    segment()
