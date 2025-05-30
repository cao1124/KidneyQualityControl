#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：kidney-quality-control 
@File    ：model.py
@IDE     ：PyCharm 
@Author  ：cao xu
@Date    ：2025/3/30 上午10:17
"""

import torch
import torch.nn as nn
from cn_clip.clip.model import VisualTransformer, BertConfig, BertModel


class MultimodalFusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 图像编码器 (使用ViT代替ResNet50)
        self.gray_image_encoder = VisualTransformer(
            input_resolution=config.image_resolution,
            patch_size=config.patch_size,
            width=config.vision_width,
            layers=config.vision_layers,
            heads=config.vision_heads,
            output_dim=config.embed_dim
        )

        self.cdfi_image_encoder = VisualTransformer(
            input_resolution=config.image_resolution,
            patch_size=config.patch_size,
            width=config.vision_width,
            layers=config.vision_layers,
            heads=config.vision_heads,
            output_dim=config.embed_dim
        )

        # 文本编码器
        bert_config = BertConfig.from_json_file(config.bert_config)
        self.text_encoder = BertModel(bert_config, add_pooling_layer=False)

        # 特征融合层
        self.gray_fc = nn.Linear(config.vision_width, config.fusion_dim)
        self.cdfi_fc = nn.Linear(config.vision_width, config.fusion_dim)
        self.text_fc = nn.Linear(bert_config.hidden_size, config.fusion_dim)

        # 融合模块
        self.fusion_layer1 = nn.Linear(config.fusion_dim, config.fusion_dim)
        self.fusion_layer2 = nn.Linear(2 * config.fusion_dim, config.fusion_dim)

        # ViT重构层
        self.vit_reconstruct = nn.Linear(
            2 * config.fusion_dim,
            config.vit_input_size[0] * config.vit_input_size[1] * config.vit_input_size[2]
        )

        # 分类ViT
        self.classifier_vit = VisualTransformer(
            input_resolution=config.vit_input_size[:2],
            patch_size=config.classifier_patch_size,
            width=config.classifier_width,
            layers=config.classifier_layers,
            heads=config.classifier_heads,
            output_dim=config.num_classes
        )

        # 激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gray_img, cdfi_img, text_input):
        # 图像特征提取
        gray_features = self.gray_image_encoder(gray_img)
        cdfi_features = self.cdfi_image_encoder(cdfi_img)

        # 文本特征提取
        text_output = self.text_encoder(**text_input)
        text_features = text_output.last_hidden_state[:, 0, :]  # [CLS] token

        # 特征映射
        feature1 = self.relu(self.gray_fc(gray_features))  # 灰阶特征 (B, 512)
        feature2 = self.relu(self.cdfi_fc(cdfi_features))  # CDFI特征 (B, 512)
        feature3 = self.relu(self.text_fc(text_features))  # 文本特征 (B, 512)

        # 第一阶段融合: CDFI + 文本
        feature4 = feature2 + feature3  # 逐元素相加

        # 第二阶段融合: (CDFI+文本) + 灰阶
        combined = torch.cat((feature4, feature1), dim=1)  # (B, 1024)
        feature5 = self.relu(self.fusion_layer1(combined))  # (B, 512)

        # 第三阶段融合: 特征5 + 特征1
        vit_input = torch.cat((feature5, feature1), dim=1)  # (B, 1024)

        # 重构为ViT输入格式
        vit_reshape = self.vit_reconstruct(vit_input)
        vit_reshape = vit_reshape.view(
            -1,
            self.config.vit_input_size[0],
            self.config.vit_input_size[1],
            self.config.vit_input_size[2]
        )  # (B, 224, 224, 64)
        vit_reshape = vit_reshape.permute(0, 3, 1, 2)  # (B, 64, 224, 224)

        # 分类ViT处理
        logits = self.classifier_vit(vit_reshape)
        return logits