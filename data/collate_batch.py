# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch

#对图像进行堆叠
def train_collate_fn(batch):
    fts, ids = zip(*batch)
    ids = torch.tensor(ids, dtype=torch.int64)
    return torch.stack(fts, dim=0), ids


def val_collate_fn(batch):
    fts, ids = zip(*batch)
    ids = torch.tensor(ids, dtype=torch.int64)
    return torch.stack(fts, dim=0), ids

