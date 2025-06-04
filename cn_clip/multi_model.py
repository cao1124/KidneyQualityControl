#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：kidney-quality-control 
@File    ：multi_model.py
@IDE     ：PyCharm 
@Author  ：cao xu
@Date    ：2025/3/30 上午10:17
"""

import torch
import torch.nn as nn
import json
from transformers import BertTokenizer, BertConfig, BertModel
from clip.model import VisualTransformer


class TextProcessor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.context_length = config.context_length

    def forward(self, texts):
        # 使用BERT tokenizer处理文本
        inputs = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.context_length,
            return_tensors='pt'
        )
        return inputs


class MultimodalFusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 图像编码器
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

        # 文本编码器 - 使用RoBERTa配置
        # 从JSON文件加载配置
        with open(config.bert_config, 'r', encoding='utf-8') as f:
            bert_config_dict = json.load(f)

        # 创建自定义配置对象
        bert_config = BertConfig(
            vocab_size=bert_config_dict.get('vocab_size', 21128),
            hidden_size=bert_config_dict.get('hidden_size', 768),
            num_hidden_layers=bert_config_dict.get('num_hidden_layers', 12),
            num_attention_heads=bert_config_dict.get('num_attention_heads', 12),
            intermediate_size=bert_config_dict.get('intermediate_size', 3072),
            hidden_act=bert_config_dict.get('hidden_act', 'gelu'),
            hidden_dropout_prob=bert_config_dict.get('hidden_dropout_prob', 0.1),
            attention_probs_dropout_prob=bert_config_dict.get('attention_probs_dropout_prob', 0.1),
            max_position_embeddings=bert_config_dict.get('max_position_embeddings', 514),
            type_vocab_size=bert_config_dict.get('type_vocab_size', 2),
            initializer_range=bert_config_dict.get('initializer_range', 0.02),
            layer_norm_eps=bert_config_dict.get('layer_norm_eps', 1e-5),
            pad_token_id=bert_config_dict.get('pad_token_id', 0)
        )

        self.text_encoder = BertModel(bert_config, add_pooling_layer=False)

        # 特征融合层
        self.gray_fc = nn.Linear(config.embed_dim, config.fusion_dim)
        self.cdfi_fc = nn.Linear(config.embed_dim, config.fusion_dim)
        self.text_fc = nn.Linear(bert_config.hidden_size, config.fusion_dim)

        # 融合模块
        self.fusion_layer1 = nn.Linear(2 * config.fusion_dim, config.fusion_dim)
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
            output_dim=config.num_classes,
            in_channels=config.vit_input_size[2]
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