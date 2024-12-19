import os
from enum import Enum

import pandas as pd
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
    data_dir = '/home/ai999/project/kidney-quality-control/clip/20241129-中山肾脏外部测试数据/复旦大学附属中山医院已完成勾图新/'
    model = torch.load('classification_model/0830-kidney-cancer-2class-0.9091.pt')
    model.eval()
    torch.cuda.empty_cache()  # 释放缓存分配器当前持有的且未占用的缓存显存
    excel_path = 'clip/20241129-中山肾脏外部测试数据/最新复旦大学附属中山医院肾肿瘤新增文本2.xlsx'
    excel_df = pd.read_excel(excel_path)
    num_list = excel_df.编号.tolist()
    lab_list = excel_df.病理诊断.tolist()
    '模型预测'
    data = []
    columns = ['病例', '图像', '医生标签', '预测结果', '预测概率']
    pred_list, label_list = [], []
    excel_file = '20241212-模型分类预测结果-单图像.xlsx'
    for root, dirs, files in os.walk(data_dir):
        for f in tqdm(files):
            # if not f.endswith('json') and os.path.exists(os.path.join(root.replace('ori', 'cancer-mask'), f)):
            if not f.endswith('json'):
                # print(os.path.join(root, f))
                image_ori = cv_read(os.path.join(root, f))
                img = Image.fromarray(image_ori).convert('RGB')
                # 器官分类、切面分类数据准备
                img = trans(img)
                img = torch.unsqueeze(img, dim=0)
                res, prob = img_classify(img, model, trans, device)
                # print(ClsCla(res).name, root.split('/')[-1])
                pred_list.append(res)
                if res == 1:
                    res = '恶'
                else:
                    res = '良'
                num = int(root.split('/')[-1].replace('-result', ''))
                idx = num_list.index(num)
                lab = lab_list[idx]
                if lab == '良':
                    label_list.append(0)
                else:
                    label_list.append(1)
                data.append((num, f, lab, res, prob))
    data_df = pd.DataFrame(data, columns=columns)
    data_df.to_excel(excel_file, index=False, engine='openpyxl')
    print('classification_report:\n{}'.format(classification_report(label_list, pred_list, digits=4)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "7"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = [0.29003, 0.29385, 0.31377], [0.18866, 0.19251, 0.19958]
    trans = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(), transforms.Normalize(mean, std)])
    test()

