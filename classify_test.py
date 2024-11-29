import os
from enum import Enum

import torch
from PIL import Image
from sklearn.metrics import classification_report
from torchvision import transforms
from tqdm import tqdm

from data_preprocess import cv_read


class ClsCla(Enum):
    良性 = 0
    恶性 = 1


def img_classify(img, model, trans, device):
    with torch.no_grad():
        output = model(img.to(device))
        pred = torch.softmax(output, dim=1).cpu()
        cls = torch.argmax(pred).numpy()
    return cls, pred[0][cls].numpy()


def test():
    data_dir = '/mnt/sdb/caoxu/dataset/十院肾囊肿/市一外部验证-crop/'
    model = torch.load('classification_model/20240910-十院肾囊肿-crop小图囊肿分类-2class-0.8980.pt')
    model.eval()
    torch.cuda.empty_cache()  # 释放缓存分配器当前持有的且未占用的缓存显存
    '模型预测'
    pred_list, label_list = [], []
    for root, dirs, files in os.walk(data_dir):
        for f in tqdm(files):
            if not f.endswith('json') and os.path.exists(os.path.join(root.replace('ori', 'cancer-mask'), f)):
                # print(os.path.join(root, f))
                image_ori = cv_read(os.path.join(root, f))
                img = Image.fromarray(image_ori).convert('RGB')
                # 器官分类、切面分类数据准备
                img = trans(img)
                img = torch.unsqueeze(img, dim=0)
                res, _ = img_classify(img, model, trans, device)
                # print(ClsCla(res).name, root.split('/')[-1])
                pred_list.append(res)
                if root.split('/')[-1] == '良性':
                    label_list.append(0)
                else:
                    label_list.append(1)
    print('classification_report:\n{}'.format(classification_report(label_list, pred_list, digits=4)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "7"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = [0.29003, 0.29385, 0.31377], [0.18866, 0.19251, 0.19958]
    trans = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(), transforms.Normalize(mean, std)])
    test()
