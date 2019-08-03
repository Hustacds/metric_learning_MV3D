# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
from torch import nn


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def euclid_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, w, l], w为特征的宽度
      y: pytorch Variable, with shape [n, l]
    Returns:
      dist: pytorch Variable, with shape [m, w, n]
    """
    m = x.size(0)
    w = x.size(1)
    n = y.size(0)
    xx = torch.pow(x, 2).sum(2, keepdim=True).expand(m, w, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, w).view(n,w,-1).expand(n,w,m).permute([2,1,0])
    dist = xx + yy - 2 *torch.matmul(x,y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    dist = torch.max(dist,dim = 1)[0]
    return dist

def cosine_dist(x,y):
    """
    Args:
      x: pytorch Variable, with shape [m, w, l], w为特征的宽度
      y: pytorch Variable, with shape [n, l]
    Returns:
      dist: pytorch Variable, with shape [m, w, n]
    """
    m = x.size(0)
    w = x.size(1)
    n = y.size(0)
    xx = torch.pow(x, 2).sum(2, keepdim=True).expand(m, w, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, w).expand(n, w, m).permute([2, 1, 0])

    dist = torch.matmul(x,y.t())/((xx*yy).clamp(min=1e-12).sqrt())
    # dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    dist = torch.max(dist, dim=1)[0]
    return dist


def hard_example_mining(dist_mat):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)
    labels = torch.arange(0,N,1)
    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    return dist_ap, dist_an


class TripletLoss(object):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, dist_type, margin=None):
        self.margin = margin
        if dist_type == 'euclid':
            self.dist = euclid_dist
        elif dist_type =='cosine':
            self.dist = cosine_dist
        else:
            print("wrong dist_type")

        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    # def __call__(self, global_feat, labels, normalize_feature=False):
    #     if normalize_feature:
    #         global_feat = normalize(global_feat, axis=-1)
    #     # dist_mat = euclidean_dist(global_feat, global_feat)
    #     dist_mat = euclidean_dist(global_feat,global_feat)
    #     dist_ap, dist_an = hard_example_mining(
    #         dist_mat, labels)
    #     y = dist_an.new().resize_as_(dist_an).fill_(1)
    #     if self.margin is not None:
    #         loss = self.ranking_loss(dist_an, dist_ap, y)
    #     else:
    #         loss = self.ranking_loss(dist_an - dist_ap, y)
    #     return loss, dist_ap, dist_an

    def __call__(self, ft_fused, ft_query, normalize_feature=False):
        if normalize_feature:
            ft_fused = normalize(ft_fused, axis = -1)
            ft_query = normalize(ft_query,axis = -1)

        dist_mat = self.dist(ft_fused,ft_query)
        dist_ap, dist_an = hard_example_mining(
            dist_mat)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_mat

