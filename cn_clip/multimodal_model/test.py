#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：kidney-quality-control 
@File    ：test.py
@IDE     ：PyCharm 
@Author  ：cao xu
@Date    ：2025/3/30 上午10:18
"""
import torch
from torch.utils.data import DataLoader
from .model import MultimodalFusionModel
from .config import ModelConfig
from cn_clip.eval.data import get_zeroshot_dataset

# 加载模型
config = ModelConfig()
model = MultimodalFusionModel(config)
model.load_state_dict(torch.load("best_fusion_model.pth"))
model = model.cuda()
model.eval()

# 测试数据集
test_dataset = get_zeroshot_dataset("your_test_dataset")  # 需实现实际数据集
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    collate_fn=custom_collate_fn  # 复用训练中的collate
)


# 评估函数
def evaluate(model, dataloader):
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            gray_img = batch['gray_img'].cuda()
            cdfi_img = batch['cdfi_img'].cuda()
            text = tokenize(batch['text']).cuda()
            labels = batch['label'].cuda()

            outputs = model(gray_img, cdfi_img, text)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


if __name__ == '__main__':

    # 运行测试
    evaluate(model, test_loader)