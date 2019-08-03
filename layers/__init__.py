# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
import torch

from .triplet_loss import TripletLoss

def make_loss(cfg,loss_type):
    triplet = TripletLoss(loss_type,cfg.SOLVER.MARGIN)
    def loss_func(ft_fused, ft_query):
        loss, dist_mat = triplet(ft_fused, ft_query)
        return loss, dist_mat
    return loss_func


