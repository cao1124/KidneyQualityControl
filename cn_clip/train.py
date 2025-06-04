#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：kidney-quality-control 
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：cao xu
@Date    ：2025/3/30 上午10:18
"""
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from multi_model import MultimodalFusionModel, TextProcessor
from config import ModelConfig
from clip.utils import image_transform
from clip.model import CLIP


def build_model(state_dict, vit=False):
    """根据状态字典构建模型结构"""
    # 提取配置参数
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    # 视觉部分配置
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len(
        [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size

    # 创建模型
    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    return model


# 加载预训练权重
def load_pretrained_weights(model, config):
    """从本地路径加载预训练权重"""
    # 模型
    model_path = config.pretrained_model_path
    checkpoint = torch.load(model_path, map_location="cpu")
    clip_state_dict = checkpoint["state_dict"]
    visual_state_dict = {}
    for k, v in clip_state_dict.items():
        if k.startswith("visual."):
            # 去掉"visual."前缀
            new_key = k[7:]
            visual_state_dict[new_key] = v

    # 加载状态字典
    model.load_state_dict(visual_state_dict, strict=False)

    # 创建预处理函数（根据模型类型设置分辨率）
    if "vit-b-16" in model_path.lower():
        input_resolution = 224
    elif "vit-l-14-336" in model_path.lower():
        input_resolution = 336
    else:
        input_resolution = 224  # 默认

    preprocess = image_transform(input_resolution)

    return model, preprocess


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
        text = self.data_df.iloc[idx]['超声表现']
        label_str = self.data_df.iloc[idx]['病理标签'].strip()
        label = self.label_map.get(label_str)

        # 加载灰阶图像
        gray_path = os.path.join(self.image_dir, case_id, f"{case_id}-灰阶.jpg")
        gray_img = self.load_image(gray_path)

        # 加载CDFI图像
        cdfi_path = os.path.join(self.image_dir, case_id, f"{case_id}-血流.jpg")
        cdfi_img = self.load_image(cdfi_path)

        return gray_img, cdfi_img, text, label

    def load_image(self, path):
        try:
            img = Image.open(path).convert('RGB')
            return self.preprocess(img)
        except Exception as e:
            # print(f"Error loading image {path}: {e}")
            # 返回空白图像作为占位符
            return self.preprocess(Image.new('RGB', (224, 224), (0, 0, 0)))


def tokenize_texts(texts, text_processor, config):
    return text_processor(texts)


def train_clip():
    print('加载配置ing...')
    config = ModelConfig()

    print('加载数据集...')
    excel_path = "/home/ai1018/project/kidney-quality-control/20250530-病例汇总.xlsx"
    image_dir = "/home/ai1018/project/kidney-quality-control/20250530-病例汇总"
    data_df = pd.read_excel(excel_path)

    print('划分数据集(8:1:1)...')
    # total_size = len(data_df)
    # train_size = int(0.8 * total_size)
    # val_size = int(0.1 * total_size)
    # test_size = total_size - train_size - val_size
    # train_df, val_df, test_df = random_split(data_df, [train_size, val_size, test_size],
    # generator=torch.Generator().manual_seed(42))

    # 划分索引
    total_size = len(data_df)
    indices = torch.randperm(total_size).tolist()
    train_end = int(0.8 * total_size)
    val_end = int(0.9 * total_size)

    # 使用iloc按索引位置获取DataFrame子集
    train_df = data_df.iloc[indices[:train_end]]
    val_df = data_df.iloc[indices[train_end:val_end]]
    test_df = data_df.iloc[indices[val_end:]]

    print('初始化模型和文本处理器...')
    model = MultimodalFusionModel(config)
    text_processor = TextProcessor(config)
    model, preprocess = load_pretrained_weights(model, config)
    model = model.cuda()

    print('创建数据集...')
    train_dataset = KidneyTumorDataset(train_df, image_dir, preprocess)
    val_dataset = KidneyTumorDataset(val_df, image_dir, preprocess)
    test_dataset = KidneyTumorDataset(test_df, image_dir, preprocess)

    print('数据加载器...')
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
    # criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01, betas=(0.9, 0.999))

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    best_val_loss = float('inf')

    print('训练循环...')
    for epoch in range(100):

        torch.cuda.empty_cache()
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
        torch.cuda.empty_cache()
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
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'config': config.__dict__
            }, "best_model.pt")
            print("  Saved best model!")

    # 测试最佳模型
    print("Testing best model on test set...")
    checkpoint = torch.load("best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_correct = 0
    test_total = 0
    torch.cuda.empty_cache()
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
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_clip()
