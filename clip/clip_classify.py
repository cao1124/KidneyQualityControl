#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : clip_classify.py
@Author  : cao xu
@Time    : 2024/3/21 9:38
"""
import os
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import CLIPProcessor, CLIPModel


def clip_classify():
    # Load CLIP model and processor
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Define image preprocessing pipeline
    image_transform = Compose([
        Resize((224, 224), interpolation=Image.BICUBIC),
        CenterCrop((224, 224)),
        ToTensor(),
        Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])

    # Define text labels
    labels = ["cat", "dog", "flower", "car"]

    def classify_image_text(image_path, text):
        # Load and preprocess image
        image = Image.open(image_path)
        image_input = image_transform(image).unsqueeze(0).to(device)

        # Tokenize text and convert to features
        inputs = processor(text, return_tensors="pt", padding=True, truncation=True).to(device)

        # Perform classification
        with torch.no_grad():
            logits_per_image, logits_per_text = clip_model(image_input, **inputs)

        # Get predicted label
        probs = torch.nn.functional.softmax(logits_per_text[0], dim=0)
        pred_label = labels[torch.argmax(probs).item()]

        return pred_label

    # Example usage
    image_path = "example.jpg"
    text = "a cute cat"
    predicted_label = classify_image_text(image_path, text)
    print("Predicted label:", predicted_label)


if __name__ == '__main__':
    clip_classify()


