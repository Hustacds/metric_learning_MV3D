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
    def __init__(self, num_query, max_rank=50, feat_norm='yes'):
        super(R1_mAP, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm

    #在每个epoch开始的时候调用
    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []
        self.v_score = []

    # feat为所有子区域的特征
    # 在每个iteration结束的时候调用
    def update(self, output):
        feat, pid, camid, visibility_score, _, _, _ = output
        self.feats.append(feat)
        self.v_score.append(visibility_score)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    # 在每个epock结束的时候调用
    def compute(self):
        v_score = torch.cat(self.v_score, dim=0)
        feats = torch.cat(self.feats, dim=0)

        if self.feat_norm == 'yes':
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        # 把生成的特征向量再拆分你为query和gallery
        # query
        qf = feats[:self.num_query]
        qv = v_score[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        gv = v_score[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

        # 计算所有图片之间的两两关系
        m, n, r = qf.shape[0], gf.shape[0],qf.shape[1]

        dismat = torch.Tensor(m, n, r).cuda()

        for i in range(r):
            dismat[:, :, i] = triplet_loss.euclidean_dist(qf[:, i, :], gf[:, i, :])

        qv_mat = qv.view(m,r,1).expand(m,r,n).permute(0,2,1)
        gv_mat = gv.view(n,r,1).expand(n,r,m).permute(2,0,1)

        v_mat = torch.mul(qv_mat,gv_mat)
        dismat = torch.sum(torch.mul(v_mat,dismat)/v_mat,dim=2)

        # dismat = torch.pow(qf, 2).sum(dim=2, keepdim=True).expand(m,3,n).permute(1,0,2)  + torch.pow(gf, 2).sum(dim=2, keepdim=True).expand(n, 3, m).permute(1,2,0)
        # dismat -= 2*torch.matmul(qf.permute(1, 0, 2), gf.permute(1, 2, 0))
        # dismat = torch.sum(dismat,dim=0)/qv.mm(gv.transpose(1, 0))
        distmat = dismat.cpu().numpy()
        print(distmat.shape)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        return cmc, mAP

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