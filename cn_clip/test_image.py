#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：kidney-quality-control 
@File    ：test_image.py
@IDE     ：PyCharm 
@Author  ：cao xu
@Date    ：2025/6/3 上午9:39 
"""
# !/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：kidney-quality-control 
@File    ：test_image.py
@IDE     ：PyCharm 
@Author  ：cao xu
"""
import torch
import json
from PIL import Image
from model import MultimodalFusionModel, TextProcessor
from config import ModelConfig
from clip import load_from_name
from transformers import BertConfig, BertModel
import argparse


# 加载模型
def load_model(checkpoint_path):
    # 加载配置
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config_dict = checkpoint['config']

    # 创建配置对象
    config = ModelConfig()
    for key, value in config_dict.items():
        setattr(config, key, value)

    # 初始化模型
    model = MultimodalFusionModel(config)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 文本处理器
    text_processor = TextProcessor(config)

    # 图像预处理
    _, preprocess = load_from_name("ViT-B-16", device="cpu")

    return model, text_processor, preprocess, config


def predict_single_case(gray_path, cdfi_path, text, model, text_processor, preprocess, config):
    # 预处理图像
    gray_img = preprocess(Image.open(gray_path).convert('RGB')).unsqueeze(0)
    cdfi_img = preprocess(Image.open(cdfi_path).convert('RGB')).unsqueeze(0)

    # 处理文本
    text_input = text_processor([text])

    # 移动到GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    gray_img = gray_img.to(device)
    cdfi_img = cdfi_img.to(device)
    text_input = {k: v.to(device) for k, v in text_input.items()}

    # 预测
    model.eval()
    with torch.no_grad():
        outputs = model(gray_img, cdfi_img, text_input)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

    # 转换预测结果
    class_names = {0: "良性", 1: "恶性", 2: "其他"}
    prediction = class_names[predicted.item()]

    return prediction, probabilities.cpu().numpy()[0]


if __name__ == '__main__':
    # python test_image.py --gray 病例汇总/1001-灰阶.jpg --cdfi 病例汇总/1001-血流.jpg --text "低回声，边界不清断，内部回声均匀，周边见少许点状彩色血流"
    parser = argparse.ArgumentParser(description='肾肿瘤多模态分类预测')
    parser.add_argument('--gray', type=str, required=True, help='灰阶图像路径')
    parser.add_argument('--cdfi', type=str, required=True, help='CDFI图像路径')
    parser.add_argument('--text', type=str, required=True, help='超声文本描述')
    parser.add_argument('--model', type=str, default='best_model.pth', help='模型路径')

    args = parser.parse_args()

    # 加载模型
    model, text_processor, preprocess, config = load_model(args.model)

    # 进行预测
    prediction, probabilities = predict_single_case(
        args.gray,
        args.cdfi,
        args.text,
        model,
        text_processor,
        preprocess,
        config
    )

    print("\n预测结果:")
    print(f"分类: {prediction}")
    print(f"概率分布: [良性: {probabilities[0]:.4f}, 恶性: {probabilities[1]:.4f}, 其他: {probabilities[2]:.4f}]")