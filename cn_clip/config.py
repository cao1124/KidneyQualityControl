class ModelConfig:
    def __init__(self):
        # 图像参数
        self.image_resolution = 224
        self.patch_size = 32

        # ViT-B-16 视觉编码器参数
        self.vision_width = 768
        self.vision_layers = 12
        self.vision_heads = 12
        self.embed_dim = 128  # 从512改为256，降低显存使用

        # 文本参数 - 使用 RoBERTa 中文模型
        self.bert_config = "clip/model_configs/RoBERTa-wwm-ext-base-chinese.json"
        self.pretrained_model_path = "pretrained_weights/clip_cn_vit-b-16.pt"
        self.text_embed_dim = 128  # 对应text_fc输出维度

        # 融合参数
        self.fusion_dim = 128  # 从512改为256，配合embed_dim
        self.vit_input_size = (224, 224, 64)
        self.classifier_patch_size = 16

        # ViT分类器参数
        self.classifier_width = 384   # 768
        self.classifier_layers = 6    # 12
        self.classifier_heads = 6     # 12

        # 训练参数
        self.num_classes = 2
        self.learning_rate = 1e-4
        self.batch_size = 2

        # 文本处理配置
        self.context_length = 77
        self.text_vocab_size = 21128
