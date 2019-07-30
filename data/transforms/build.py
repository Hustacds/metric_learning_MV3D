# encoding: utf-8

#定义训练数据的变换

import torchvision.transforms as T
from .transforms import RandomErasing,RandomCrop

def build_transforms(cfg, is_train=True):
    normalize_transform = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    partial_crop = RandomCrop(cfg.MODEL.CROP_POS,cfg.MODEL.CROP_RATIO,cfg.MODEL.R_REGION,cfg.MODEL.C_REGION,cfg.INPUT.SIZE_TRAIN)
    if is_train:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN),
            # T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            # T.Pad(cfg.INPUT.PADDING),
            # T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            normalize_transform,

            #随机擦除，但并非裁剪，而是改为背景色
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])
    else:
        transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            normalize_transform
        ])

    return transform,partial_crop
