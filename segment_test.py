import copy
import os
import random
import cv2
import segmentation_models_pytorch
import torch
import numpy as np
from tqdm import tqdm

from data_preprocess import cv_read, cv_write
from segment_util import get_iou, get_f1, add_weighted, combine_image


def test():
    data_dir = '/mnt/sdb/caoxu/dataset/十院肾囊肿/市一外部验证-ori/'
    save_dir = '/mnt/sdb/caoxu/dataset/十院肾囊肿/市一外部验证-pred/'
    os.makedirs(save_dir, exist_ok=True)
    '模型加载'
    model = torch.load('mass-segment/20240910-十院肾囊肿-囊肿分割-efficientnet-b7-0.8656.pt')
    model.eval()
    torch.cuda.empty_cache()  # 释放缓存分配器当前持有的且未占用的缓存显存
    '模型预测'
    iou_list, dice_list = [], []
    for root, dirs, files in os.walk(data_dir):
        for f in files:
            if not f.endswith('json') and os.path.exists(os.path.join(root.replace('ori', 'cancer-mask'), f)):
                print(os.path.join(root, f))
                image_ori = cv_read(os.path.join(root, f))
                gt_mask = cv_read(os.path.join(root.replace('ori', 'cancer-mask'), f))
                [orig_h, orig_w, _] = image_ori.shape
                image = cv2.resize(image_ori, (512, 512), interpolation=cv2.INTER_NEAREST)
                image = image / 255.0
                image = image.transpose(2, 0, 1).astype('float32')
                image_tensor = torch.from_numpy(image).to(device).unsqueeze(0)
                with torch.no_grad():
                    pr_mask = model(image_tensor).squeeze().cpu().numpy().round()

                pr_mask[pr_mask < 0.5] = 0
                pr_mask[pr_mask >= 0.5] = 255
                if pr_mask.shape != image_ori.shape:
                    pr_mask = cv2.resize(pr_mask, (orig_w, orig_h), cv2.INTER_NEAREST)
                    _, pr_mask = cv2.threshold(pr_mask, 1, 255, cv2.THRESH_BINARY)

                if np.sum(gt_mask) == 0 and np.sum(pr_mask) == 0:
                    iou = 1
                    dice = 1
                else:
                    iou = get_iou(gt_mask, pr_mask)
                    dice = get_f1(gt_mask, pr_mask)
                iou = np.round(iou, 4)
                dice = np.round(dice, 4)
                iou_list.append(iou)
                dice_list.append(dice)
                save_full_path = os.path.join(save_dir, f)
                img_gt = add_weighted(image_ori, gt_mask.astype('uint8'), 'BG')
                img_pred = add_weighted(image_ori, pr_mask.astype('uint8'), 'GR')
                img_gt_pred = combine_image(img_gt, img_pred)
                cv_write(save_full_path, img_gt_pred)
    print("\tAll Mean Dice:", np.average(dice_list))
    print("\tAll Mean IoU:", np.average(iou_list))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "7"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test()
