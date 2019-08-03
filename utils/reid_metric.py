# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import torch
from ignite.metrics import Metric

from data.datasets.eval_reid import eval_func
from .re_ranking import re_ranking
from layers import triplet_loss

# 相似度的度量
class R1_mAP(Metric):
    def __init__(self, loss_type, max_rank=50, feat_norm='yes'):
        super(R1_mAP, self).__init__()
        self.loss_type = loss_type

        if loss_type == 'euclid':
            self.dist_fn = triplet_loss.euclid_dist
        elif loss_type =='cosine':
            self.dist_fn = triplet_loss.cosine_dist

        self.max_rank = max_rank
        self.feat_norm = feat_norm

    #在每个epoch开始的时候调用
    def reset(self):
        self.ft_fused = []
        self.ft_query = []
        self.target = []


    # feat为所有子区域的特征
    # 在每个iteration结束的时候调用
    def update(self, output):
        batch_ft_fused, batch_ft_query, batch_target, _, _ = output
        self.ft_fused.append(batch_ft_fused)
        self.ft_query.append(batch_ft_query)
        self.target.extend(np.asarray(batch_target))

    # 在每个epock结束的时候调用
    def compute(self):

        gf =  torch.cat(self.ft_fused,dim=0)
        qf =  torch.cat(self.ft_query,dim =0)
        g_pids = np.asarray(self.target)
        q_pids = np.asarray(self.target)

        # if self.feat_norm == 'yes':
        #     print("The test feature is normalized")
        #     gf = torch.nn.functional.normalize(gf, dim=2, p=2)
        #     qf = torch.nn.functional.normalize(qf, dim=1, p=2)

       # 计算所有特征之间的两两关系
        dismat = self.dist_fn(gf,qf)
        dismat = dismat.t()
        distmat = dismat.cpu().numpy()

        print(distmat.shape)

        num_q, num_g = distmat.shape
        if num_g < self.max_rank:
            max_rank = num_g
            print("Note: number of gallery samples is quite small, got {}".format(num_g))
        else:
            max_rank = self.max_rank
        indices = np.argsort(distmat, axis=1)  # distmat的size为num_q * num_g,  对gallery feature与每个query feature之间的距离进行排序
        matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)  # np.newaxis是增加一维

        # compute cmc curve for each query
        all_cmc = []
        all_AP = []
        num_valid_q = 0.  # number of valid query
        for q_idx in range(num_q):
            # get query pid and camid
            # compute cmc curve
            # binary vector, positions with value 1 are correct matches
            orig_cmc = matches[q_idx]
            if not np.any(orig_cmc):
                # this condition is true when query identity does not appear in gallery
                continue

            cmc = orig_cmc.cumsum()
            cmc[cmc > 1] = 1  # 这里是？

            all_cmc.append(cmc[:max_rank])
            num_valid_q += 1.

            # compute average precision
            # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
            num_rel = orig_cmc.sum()  # 预测匹配上的数量
            tmp_cmc = orig_cmc.cumsum()
            tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
            AP = tmp_cmc.sum() / num_rel
            all_AP.append(AP)

        assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

        all_cmc = np.asarray(all_cmc).astype(np.float32)
        all_cmc = all_cmc.sum(0) / num_valid_q
        mAP = np.mean(all_AP)

        return all_cmc, mAP

#计算jaccard相似度作为距离
class R1_mAP_reranking(Metric):
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP_reranking, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        # m, n = qf.shape[0], gf.shape[0]
        #distmat = torch.pow(qf[:, 0, :], 2).sum(dim=1, keepdim=True).expand(m, n) + torch.pow(gf[:, 0, :], 2).sum(dim=1,keepdim=True).expand(n, m).t()
        # distmat.addmm_(1, -2, qf, gf.t())
        # distmat = distmat.cpu().numpy()
        print("Enter reranking")
        distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

        return cmc, mAP