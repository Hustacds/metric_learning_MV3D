# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import DataLoader

from .collate_batch import train_collate_fn, val_collate_fn
from .datasets import init_dataset, ImageDataset
from .samplers import RandomIdentitySampler, RandomIdentitySampler_alignedreid  # New add by gu
from .transforms import build_transforms

def make_data_loader(cfg):
    train_transforms,partial_crop = build_transforms(cfg, is_train=True)
    val_transforms,partial_crop = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    if len(cfg.DATASETS.NAMES) == 1:
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    else:
        # TODO: add multi dataset to train
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    num_classes = dataset.num_train_pids
    train_set = ImageDataset(dataset.train, train_transforms, partial_crop)
    #如果只做分类，用softmax就好，如果需要计算triplet，必须要用RandomIdentitySampl1er
    if cfg.DATALOADER.SAMPLER == 'softmax':  #按id来
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:   #按子batch来
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            num_workers=num_workers, collate_fn=train_collate_fn
        )

    # val_set = ImageDataset(dataset.train)
    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms,partial_crop)
    val_loader = DataLoader(
        val_set, batch_size= cfg.SOLVER.IMS_PER_BATCH, sampler=RandomIdentitySampler(dataset.query + dataset.gallery, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
        num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    # val_loader = DataLoader(
    #     val_set, batch_size=cfg.TEST.IMS_PER_BATCH,
    #     shuffle=False,
    #     collate_fn=val_collate_fn
    # )

    return train_loader, val_loader, len(dataset.query), num_classes



