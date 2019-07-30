# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch

#对图像进行堆叠
def train_collate_fn(batch):
    imgs, pids, _, _, imgs_partial, region_label = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    # print(imgs)
    # print(pids)
    # print(imgs_partial)
    # print(region_label)
    return torch.stack(imgs, dim=0), pids, torch.stack(imgs_partial,dim =0) ,torch.stack(region_label,dim=0)


def val_collate_fn(batch):
    imgs, pids, camids, _, imgs_partial, region_label= zip(*batch)
    # print("val_collate_fn===========",img_path)
    # print("val_collate_fn===========",pids)
    return torch.stack(imgs, dim=0), pids, camids, torch.stack(imgs_partial,dim =0) ,torch.stack(region_label,dim=0)
