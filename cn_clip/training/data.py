from math import ceil
import os
import logging
from pathlib import Path
import json
from PIL import Image
import base64
from io import BytesIO
from dataclasses import dataclass

import lmdb
import pickle

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
from timm.data import create_transform

from clip import _tokenizer
from clip import tokenize


def _convert_to_rgb(image):
    return image.convert('RGB')


def _preprocess_text(text):
    # adapt the text to Chinese BERT vocab
    text = text.lower().replace("“", "\"").replace("”", "\"")
    return text


class LMDBDataset(Dataset):
    def __init__(self, lmdb_path, split="val", max_txt_length=64, use_augment=False, resolution=224):
        self.lmdb_path = lmdb_path

        # assert LMDB directories exist
        assert os.path.isdir(lmdb_path), "The LMDB directory {} of {} split does not exist!".format(lmdb_path, split)
        lmdb_pairs = os.path.join(lmdb_path, "pairs")
        assert os.path.isdir(lmdb_pairs), "The LMDB directory {} of {} image-text pairs does not exist!".format(lmdb_pairs, split)
        lmdb_imgs = os.path.join(lmdb_path, "imgs")
        assert os.path.isdir(lmdb_imgs), "The LMDB directory {} of {} image base64 strings does not exist!".format(lmdb_imgs, split)

        # open LMDB files
        self.env_pairs = lmdb.open(lmdb_pairs, readonly=True, create=False, lock=False, readahead=False, meminit=False)
        self.txn_pairs = self.env_pairs.begin(buffers=True)
        self.env_imgs = lmdb.open(lmdb_imgs, readonly=True, create=False, lock=False, readahead=False, meminit=False)
        self.txn_imgs = self.env_imgs.begin(buffers=True)

        # fetch number of pairs and images
        self.number_samples = int(self.txn_pairs.get(key=b'num_samples').tobytes().decode('utf-8'))
        self.number_images = int(self.txn_imgs.get(key=b'num_images').tobytes().decode('utf-8'))
        logging.info("{} LMDB file contains {} images and {} pairs.".format(split, self.number_images, self.number_samples))

        super(LMDBDataset, self).__init__()

        # the self.dataset_len will be edited to a larger value by calling pad_dataset()
        self.dataset_len = self.number_samples
        self.global_batch_size = 1 # will be modified to the exact global_batch_size after calling pad_dataset()

        self.split = split
        self.max_txt_length = max_txt_length        

        self.use_augment = use_augment
        self.transform = self._build_transform(resolution)

        # self.text_tags = {
        #     '这是一张可见乳腺的超声检查图片':0,
        #     '这是一张可见卵巢的超声检查图片':1,
        #     '这是一张可见胰腺的超声检查图片':2,
        #     '这是一张可见颈动脉的超声检查图片':3,
        #     '这是一张可见肾脏的超声检查图片':4,
        #     '这是一张可见肾脏、脾脏的超声检查图片':5,
        #     '这是一张可见胆囊、肝脏的超声检查图片':6,
        #     '这是一张可见前列腺的超声检查图片':7,
        #     '这是一张可见膀胱、子宫的超声检查图片':8,
        #     '这是一张可见其他未知器官的超声检查图片':9,
        #     '这是一张可见肝脏的超声检查图片':10,
        #     '这是一张可见子宫的超声检查图片':11,
        #     '这是一张可见脾脏的超声检查图片':12,
        #     '这是一张可见甲状腺的超声检查图片':13,
        #     '这是一张可见膀胱、前列腺的超声检查图片':14,
        #     '这是一张可见膀胱、子宫、卵巢的超声检查图片':15,
        #     '这是一张可见心脏的超声检查图片':16,
        #     '这是一张可见颈部淋巴的超声检查图片':17,
        #     '这是一张可见肾脏、肝脏的超声检查图片':18,
        #     '这是一张可见胆囊的超声检查图片':19,
        #     '这是一张可见子宫、卵巢的超声检查图片':20,
        #     '这是一张可见膀胱的超声检查图片':21,
        #     '这是一张可见膀胱、卵巢的超声检查图片':22,
        #     '这是一张可见甲状腺、颈部淋巴的超声检查图片':23,
        # }

        self.text_tags = {
            '这是一张可见脾脏长轴切面的超声检查图片':0,
            '这是一张可见胆囊长轴切面的超声检查图片':1,
            '这是一张可见肝右肾矢状切面、右肾长轴切面的超声检查图片':2,
            '这是一张可见胰腺长轴切面的超声检查图片':3,
            '这是一张可见其他的超声检查图片':4,
            '这是一张可见右肋缘下经右肝膈顶斜切面的超声检查图片':5,
            '这是一张可见经腹主动脉矢状切面的超声检查图片':6,
            '这是一张可见左肾长轴切面的超声检查图片':7,
            '这是一张可见经第一肝门右肝斜切面彩色多普勒血流图、胆囊长轴切面的超声检查图片':8,
            '这是一张可见右肾长轴切面的超声检查图片':9,
            '这是一张可见剑突下横切面的超声检查图片':10,
            '这是一张可见经第二肝门斜断面的超声检查图片':11,
            '这是一张可见经第一肝门右肝斜切面彩色多普勒血流图的超声检查图片':12,
            '这是一张可见脾脏长轴切面、左肾长轴切面的超声检查图片':13,
            '这是一张可见胆总管上段长轴切面图的超声检查图片':14,
            '这是一张可见胆囊长轴切面、胆总管上段长轴切面图的超声检查图片':15,
            '这是一张可见肝右肾矢状切面的超声检查图片':16,
            '这是一张可见经第一肝门右肝斜切面彩色多普勒血流图、胆总管上段长轴切面图的超声检查图片':17,
            '这是一张可见经第一肝门右肝斜切面彩色多普勒血流图、胆囊长轴切面、胆总管上段长轴切面图的超声检查图片':18,
            '这是一张可见剑突下横切面、胰腺长轴切面的超声检查图片':19,
        }

    def _build_transform(self, resolution):
        if self.split == "train" and self.use_augment:
            transform = create_transform(
                             input_size=resolution,
                             scale=(0.9, 1.0),
                             is_training=True,
                             color_jitter=None,
                             auto_augment='original',
                             interpolation='bicubic',
                             mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711),
                         )
            transform = Compose(transform.transforms[:-3] + [_convert_to_rgb] + transform.transforms[-3:])
        else:
            transform = Compose([
                Resize((resolution, resolution), interpolation=InterpolationMode.BICUBIC),
                _convert_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        return transform

    def __del__(self):
        if hasattr(self, 'env_pairs'):
            self.env_pairs.close()
        if hasattr(self, 'env_imgs'):
            self.env_imgs.close()

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        sample_index = index % self.number_samples

        pair = pickle.loads(self.txn_pairs.get("{}".format(sample_index).encode('utf-8')).tobytes())
        image_id, text_id, raw_text = pair

        image_b64 = self.txn_imgs.get("{}".format(image_id).encode('utf-8')).tobytes()
        image_b64 = image_b64.decode(encoding="utf8", errors="ignore")
        image = Image.open(BytesIO(base64.urlsafe_b64decode(image_b64))) # already resized
        image = self.transform(image)

        text = tokenize([_preprocess_text(raw_text)], context_length=self.max_txt_length)[0]
        eos_index = text.numpy().tolist().index(_tokenizer.vocab['[SEP]'])

        text_tag = self.text_tags[raw_text]
        # print(raw_text)
        return image, text, eos_index, text_tag


def pad_dataset(dataset, global_batch_size):
    # edit dataset.__len__() of the dataset
    dataset.dataset_len = ceil(dataset.dataset_len / global_batch_size) * global_batch_size
    dataset.global_batch_size = global_batch_size


def fetch_resolution(vision_model):
    # fetch the resolution from the vision model config
    vision_model_config_file = Path(__file__).parent.parent / f"clip/model_configs/{vision_model.replace('/', '-')}.json"
    with open(vision_model_config_file, 'r') as fv:
        model_info = json.load(fv)
    return model_info["image_resolution"]


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler
    dataset: LMDBDataset
    epoch_id: int


def get_dataset(args, is_train, max_txt_length=64, epoch_id=0):
    if is_train:
        db_path = args.train_data
    else:
        db_path = args.val_data
    assert db_path is not None

    dataset = LMDBDataset(
        db_path, 
        split="train" if is_train else "val",
        max_txt_length=max_txt_length,
        use_augment=args.use_augment if is_train else False,
        resolution=fetch_resolution(args.vision_model),
    ) 

    # pad the dataset splits using the beginning samples in the LMDB files
    # to make the number of samples enough for a full final global batch
    batch_size = args.batch_size if is_train else args.valid_batch_size
    global_batch_size = batch_size * torch.distributed.get_world_size()
    pad_dataset(dataset, global_batch_size)

    num_samples = dataset.dataset_len
    # Update in 22.12.11: We have changed the **validation** dataset sampler during finetuning
    # from sequential to shuffled (in a determistic order between experiments and epochs). 
    # This is to avoid there being one text matching multiple images (or vice versa) in a local batch
    # which will affect the correctness of computing the validation in-batch accuracy.
    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed)
    sampler.set_epoch(epoch_id if is_train else 0)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=False,
        num_workers=args.num_workers if is_train else args.valid_num_workers,
        sampler=sampler,
    )

    dataloader.num_samples = num_samples
    assert num_samples % dataset.global_batch_size == 0
    dataloader.num_batches = num_samples // dataset.global_batch_size

    return DataInfo(dataloader, sampler, dataset, epoch_id)


def get_data(args, epoch_id=0, max_txt_length=64):
    data = {}

    if args.train_data:
        data["train"] = get_dataset(
            args, 
            is_train=True,  
            max_txt_length=max_txt_length, 
            epoch_id=epoch_id)

    if args.val_data:
        data["val"] = get_dataset(
            args, 
            is_train=False, 
            max_txt_length=max_txt_length, 
            epoch_id=epoch_id)

    return data
