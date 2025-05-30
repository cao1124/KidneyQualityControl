#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : clip_classify.py
@Author  : cao xu
@Time    : 2024/3/21 9:38
"""
import os

import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm

import clip
from clip_train import load_pretrian_model


def clip_classify():
    # Load CLIP model and processor
    os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # clip_model, processor = load_pretrian_model('20240422-clip-resnet-classify-ViT-B32-model.pt', device)
    model_clip, _ = load_pretrian_model('ViT-B/32', device)
    fc_layer = torch.nn.Linear(1024, 3 * 224 * 224).to(device)
    clip_model = torch.load('20240821-clip-resnext50-classify-0.9863.pt')
    clip_model.eval()
    # Define image preprocessing pipeline
    image_transform = Compose([
        Resize((224, 224), interpolation=Image.BICUBIC),
        CenterCrop((224, 224)),
        ToTensor(),
        Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    # Define text labels
    # labels = ["cat", "dog", "flower", "car"]
    labels = ["benign", "malignant"]

    def classify_image_text(image_path, text):
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_input = image_transform(image).unsqueeze(0).to(device)
        text_input = clip.tokenize(text).to(device)

        # Perform classification
        with torch.no_grad():
            gray_img_feature, text_feature = model_clip(image_input, text_input)
            blood_img_feature, text_feature = model_clip(image_input, text_input)
            img_with_text = torch.cat((gray_img_feature, blood_img_feature, text_feature), dim=1).float()
            img_with_text = img_with_text.view(img_with_text.size(0), -1, 1, 1)
            # 使用一个全连接层将特征映射到一个更高维度的空间
            img_with_text_mapped = fc_layer(img_with_text.view(-1, 1024)).view(1, 3, 224, 224)
            output = clip_model(img_with_text_mapped)

        # Get predicted label
        pred_label = labels[output.argmax(dim=1).cpu().numpy()[0]]

        return pred_label

    # Example usage
    img_dir = '20241129-中山肾脏外部测试数据/复旦大学附属中山医院已完成勾图'
    df = {'image': [], 'caption': [], 'label': []}
    excel_path = '20241129-中山肾脏外部测试数据/复旦大学附属中山医院肾肿瘤新增文本-EN.xlsx'
    excel_df = pd.read_excel(excel_path)
    num_list = excel_df.iloc[:, 0].tolist()
    for c in os.listdir(img_dir):
        for p in os.listdir(os.path.join(img_dir, c)):
            idx = num_list.index(int(p.replace('-result', '')))
            for n in os.listdir(os.path.join(img_dir, c, p)):
                if not n.endswith('.json'):
                    df['image'].append(os.path.join(img_dir, c, p, n))
                    if c == 'benign':
                        df['label'].append(0)
                    else:
                        df['label'].append(1)
                    description = excel_df.iloc[idx][1]
                    if excel_df.iloc[idx][3] == '女':
                        sex = 'woman'
                    else:
                        sex = 'man'
                    year = int(excel_df.iloc[idx][4])
                    loc = excel_df.iloc[idx][5]
                    if loc == '左':
                        located = 'left kidney'
                    else:
                        located = 'right kidney'
                    pa = excel_df.iloc[idx][6]
                    if pa == '上':
                        part = 'upper kidney location'
                    elif pa == '中':
                        part = 'middle kidney location'
                    else:
                        part = 'lower kidney location'
                    maximum = int(excel_df.iloc[idx][7])
                    df['caption'].append(f"A photo of a kidney cancer image showing a tumor with {description} in "
                                         f"a {year}-year-old {sex}, with a maximum diameter of {maximum} mm, located "
                                         f"on the {located} kidney, in the {part}.")
    columns = ['图像输入', '文本输入', '预测结果', '医生标签']
    data = []
    excel_file = r'20250317-clip预测结果.xlsx'
    for i in range(len(df['image'])):
        image_path = df['image'][i]
        text = df['caption'][i]
        label = df['label'][i]
        res = classify_image_text(image_path, text)
        print("img: {}, caption: {}".format(image_path, text), "Predicted res:", res, 'Label:', label)
        data.append((image_path, text, res, label))
    data_df = pd.DataFrame(data, columns=columns)
    data_df.to_excel(excel_file, index=False, engine='openpyxl')


if __name__ == '__main__':
    clip_classify()


