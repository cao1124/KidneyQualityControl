#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：kidney-quality-control 
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：cao xu
@Date    ：2025/3/30 上午10:18
"""
import json
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from multi_model import MultimodalFusionModel, TextProcessor
from config import ModelConfig
from clip import load_from_name
from transformers import BertConfig, BertModel


# 加载预训练权重
def load_pretrained_weights(model, config):
    # 加载Chinese-CLIP预训练权重
    clip_model, _ = load_from_name("ViT-B-16", device=device)

    # 加载图像编码器权重
    model.gray_image_encoder.load_state_dict(clip_model.visual.state_dict())
    model.cdfi_image_encoder.load_state_dict(clip_model.visual.state_dict())

    # 加载文本编码器权重
    with open(config.bert_config, 'r', encoding='utf-8') as f:
        bert_config = json.load(f)
    model.text_encoder = BertModel(BertConfig(**bert_config))
    model.text_encoder.load_state_dict(clip_model.textual.state_dict())

    return model


# 自定义数据集类
class KidneyTumorDataset(Dataset):
    def __init__(self, data_df, image_dir, preprocess):
        """
        data_df: 包含病例信息的DataFrame
        image_dir: 图像存储目录
        preprocess: 图像预处理函数
        """
        self.data_df = data_df
        self.image_dir = image_dir
        self.preprocess = preprocess
        self.label_map = {"良": 0, "恶": 1}  # 根据实际标签修改

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        case_id = str(self.data_df.iloc[idx]['编号'])
        text = self.data_df.iloc[idx]['超声文本描述']
        label_str = self.data_df.iloc[idx]['病理诊断'].strip()
        label = self.label_map.get(label_str, 2)  # 默认为"其他"类别

        # 加载灰阶图像
        gray_path = os.path.join(self.image_dir, f"{case_id}-灰阶.jpg")
        gray_img = self.load_image(gray_path)

        # 加载CDFI图像
        cdfi_path = os.path.join(self.image_dir, f"{case_id}-血流.jpg")
        cdfi_img = self.load_image(cdfi_path)

        return gray_img, cdfi_img, text, label

    def load_image(self, path):
        try:
            img = Image.open(path).convert('RGB')
            return self.preprocess(img)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # 返回空白图像作为占位符
            return self.preprocess(Image.new('RGB', (224, 224), (0, 0, 0)))


def tokenize_texts(texts, text_processor, config):
    return text_processor(texts)


def train_clip():
    # 加载配置
    config = ModelConfig()

    # 加载数据集
    excel_path = "病例汇总.xlsx"
    image_dir = "病例汇总"
    data_df = pd.read_excel(excel_path)

    # 划分数据集 (8:1:1)
    total_size = len(data_df)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_df, val_df, test_df = random_split(data_df, [train_size, val_size, test_size],
                                             generator=torch.Generator().manual_seed(42))

    # 初始化模型和文本处理器
    model = MultimodalFusionModel(config)
    text_processor = TextProcessor(config)
    model, preprocess = load_pretrained_weights(model, config)
    model = model.cuda()

    # 创建数据集
    train_dataset = KidneyTumorDataset(train_df, image_dir, preprocess)
    val_dataset = KidneyTumorDataset(val_df, image_dir, preprocess)
    test_dataset = KidneyTumorDataset(test_df, image_dir, preprocess)

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2
    )

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    best_val_loss = float('inf')

    # 训练循环
    for epoch in range(100):
        model.train()
        total_loss = 0.0

        for gray_img, cdfi_img, texts, labels in train_loader:
            gray_img = gray_img.cuda()
            cdfi_img = cdfi_img.cuda()
            labels = labels.cuda()

            # 文本处理
            text_input = tokenize_texts(texts, text_processor, config)
            text_input = {k: v.cuda() for k, v in text_input.items()}

            # 前向传播
            optimizer.zero_grad()
            outputs = model(gray_img, cdfi_img, text_input)
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for gray_img, cdfi_img, texts, labels in val_loader:
                gray_img = gray_img.cuda()
                cdfi_img = cdfi_img.cuda()
                labels = labels.cuda()

                text_input = tokenize_texts(texts, text_processor, config)
                text_input = {k: v.cuda() for k, v in text_input.items()}

                outputs = model(gray_img, cdfi_img, text_input)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        # 更新学习率
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch + 1}/{100}:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'config': config.__dict__
            }, "best_model.pth")
            print("  Saved best model!")

    # 测试最佳模型
    print("Testing best model on test set...")
    checkpoint = torch.load("best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for gray_img, cdfi_img, texts, labels in test_loader:
            gray_img = gray_img.cuda()
            cdfi_img = cdfi_img.cuda()
            labels = labels.cuda()

            text_input = tokenize_texts(texts, text_processor, config)
            text_input = {k: v.cuda() for k, v in text_input.items()}

            outputs = model(gray_img, cdfi_img, text_input)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_acc = test_correct / test_total
    print(f"Test Accuracy: {test_acc:.4f}")


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_clip()
