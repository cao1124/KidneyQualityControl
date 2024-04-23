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
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, processor = load_pretrian_model(r'D:\PycharmProjects\kidney-quality-control\clip\clip-model\20240422-clip-classify-model.pt', device)

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
            logits_per_image, logits_per_text = clip_model(image_input, text_input)

        # Get predicted label
        probs = torch.nn.functional.softmax(logits_per_text[0], dim=0)
        pred_label = labels[torch.argmax(probs).item()]

        return pred_label

    # Example usage
    img_dir = r'E:\med_dataset\kidney_dataset\kidney-zhongshan\20240312-kidney-5fold\fold4'
    img_list, label_list = [], []
    excel_path = '复旦中山医院肾肿瘤病理编号1-600共508例.csv'
    excel_df = pd.read_csv(excel_path, encoding='utf-8')
    num_list = excel_df.iloc[:, 0].tolist()
    for c in os.listdir(img_dir):
        for p in os.listdir(os.path.join(img_dir, c)):
            idx = num_list.index(int(p))
            for n in os.listdir(os.path.join(img_dir, c, p)):
                img_list.append(os.path.join(img_dir, c, p, n))
                if c == '0':
                    cla = 'benign'
                else:
                    cla = 'malignant'
                if excel_df.iloc[idx][3] == '女':
                    sex = 'woman'
                else:
                    sex = 'man'
                year = int(excel_df.iloc[idx][4])
                label_list.append("a photo of {} kidney cancer image in a {}-year-old {}.".format(cla, year, sex))
    columns = ['图像输入', '文本标签', '预测结果']
    data = []
    excel_file = r'20240422-clip预测结果.xlsx'
    for i in tqdm(range(len(img_list))):
        image_path = img_list[i]
        text = label_list[i]
        predicted_label = classify_image_text(image_path, text)
        # print("img: {}, label: {}".format(image_path, text), "Predicted label:", predicted_label)
        data.append((image_path, text, predicted_label))
    data_df = pd.DataFrame(data, columns=columns)
    data_df.to_excel(excel_file, index=False, engine='openpyxl')


if __name__ == '__main__':
    clip_classify()


