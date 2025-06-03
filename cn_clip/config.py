#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：kidney-quality-control 
@File    ：config.py
@IDE     ：PyCharm 
@Author  ：cao xu
@Date    ：2025/3/30 上午10:17
"""


class ModelConfig:
    def __init__(self):
        # 图像参数
        self.image_resolution = 224
        self.patch_size = 32

        # 文本参数
        self.bert_config = "cn_clip/clip/model_configs/bert_base_chinese.json"

        # 融合参数
        self.fusion_dim = 512
        self.vit_input_size = (224, 224, 64)  # H, W, C
        self.classifier_patch_size = 16

        # ViT分类器参数
        self.classifier_width = 768
        self.classifier_layers = 12
        self.classifier_heads = 12

        # 训练参数
        self.num_classes = 3  # 根据实际类别修改
        self.learning_rate = 1e-5
        self.batch_size = 32

        # 新增文本处理配置
        self.context_length = 77  # 文本最大长度
        self.text_vocab_size = 21128  # 中文BERT词表大小