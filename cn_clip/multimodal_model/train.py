#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：kidney-quality-control 
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：cao xu
@Date    ：2025/3/30 上午10:18
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .model import MultimodalFusionModel
from .config import ModelConfig
from cn_clip.clip import load_from_name
from cn_clip.eval.data import get_zeroshot_dataset


# 加载预训练权重
def load_pretrained_weights(model):
    # 加载Chinese-CLIP预训练权重
    clip_model, _ = load_from_name("ViT-B-16", device="cpu")

    # 加载图像编码器权重
    model.gray_image_encoder.load_state_dict(clip_model.visual.state_dict())
    model.cdfi_image_encoder.load_state_dict(clip_model.visual.state_dict())

    # 加载文本编码器权重
    model.text_encoder.load_state_dict(clip_model.textual.state_dict())

    return model

def custom_collate_fn(batch):
    gray_imgs, cdfi_imgs, texts, labels = zip(*batch)
    return {
        'gray_img': torch.stack(gray_imgs),
        'cdfi_img': torch.stack(cdfi_imgs),
        'text': texts,  # 需要单独进行tokenize
        'label': torch.tensor(labels)
    }


if __name__ == '__main__':

    # 加载配置
    config = ModelConfig()

    # 初始化模型
    model = MultimodalFusionModel(config)

    model = load_pretrained_weights(model)
    model = model.cuda()

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # 数据集加载
    train_dataset = get_zeroshot_dataset("your_train_dataset")  # 需实现实际数据集
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        collate_fn=custom_collate_fn,
        shuffle=True
    )

    # 训练循环
    for epoch in range(10):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            gray_img = batch['gray_img'].cuda()
            cdfi_img = batch['cdfi_img'].cuda()
            text = tokenize(batch['text']).cuda()  # 实现文本tokenize
            labels = batch['label'].cuda()

            # 前向传播
            optimizer.zero_grad()
            outputs = model(gray_img, cdfi_img, text)
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")
        torch.save(model.state_dict(), f"fusion_model_epoch_{epoch + 1}.pt")