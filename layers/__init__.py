# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
import torch

from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth,TripletLoss_VPM
from .cluster_loss import ClusterLoss

#定义VPM损失函数，损失函数由loss_r,loss_id,loss_tri三部分组成
def make_VPM_loss(cfg, num_classes):
    sampler = cfg.DATALOADER.SAMPLER
    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        triplet = TripletLoss_VPM(cfg.SOLVER.MARGIN)  # triplet loss
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'cluster':
        cluster = ClusterLoss(cfg.SOLVER.CLUSTER_MARGIN, True, True, cfg.SOLVER.IMS_PER_BATCH // cfg.DATALOADER.NUM_INSTANCE, cfg.DATALOADER.NUM_INSTANCE)
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_cluster':
        triplet = TripletLoss_VPM(cfg.SOLVER.MARGIN)  # triplet loss
        cluster = ClusterLoss(cfg.SOLVER.CLUSTER_MARGIN, True, True, cfg.SOLVER.IMS_PER_BATCH // cfg.DATALOADER.NUM_INSTANCE, cfg.DATALOADER.NUM_INSTANCE)
    else:
        print('expected METRIC_LOSS_TYPE should be triplet, cluster, triplet_cluster'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif sampler == 'triplet':
        def loss_func(score, feat, target):
            return triplet(feat, target)[0]
    elif sampler == 'softmax_triplet':
        # changed by ds for basic vpm
        def loss_func_vpm(feature_map, probability_map, visibility_score, feature_region, class_score,target,region_label):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':

                v_min = torch.min(torch.min(region_label[:,:,:],dim = 2)[0],dim=1)[0]
                v_max = torch.max(torch.max(region_label[:,:,:],dim = 2)[0],dim=1)[0]

                m = feature_region.shape[0]
                r = feature_region.shape[1]

                #计算id损失
                loss_id = torch.zeros(m).cuda()
                for i_batch in range(m):
                    V = torch.arange(v_min[i_batch],v_max[i_batch]+1)
                    s = class_score.permute(0, 2, 1)[:,:, V]
                    t =target.view(m, 1).expand(m, r)[:, V]
                    loss_id[i_batch]= F.cross_entropy(s, t)
                l_id = torch.mean(loss_id)

                l_r = F.cross_entropy(probability_map, region_label)
                l_tri = triplet(feature_region,target,region_label)

                score = torch.sum(class_score,dim = 1)

                return l_r + l_id + l_tri[0],score,l_r,l_id,l_tri[0]
            else:
                print('expected METRIC_LOSS_TYPE should be triplet, cluster, triplet_cluster，'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    else:
        print('expected sampler should be softmax, triplet or softmax_triplet, '
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func_vpm



def make_VPM_loss_val(cfg):
    triplet = TripletLoss_VPM(cfg.SOLVER.MARGIN)
    def loss_func_vpm_val( probability_map, feature_region,  target,
                      region_label):
        if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':

            # v_min = torch.min(torch.min(region_label[:, :, :], dim=2)[0], dim=1)[0]
            # v_max = torch.max(torch.max(region_label[:, :, :], dim=2)[0], dim=1)[0]
            # m = feature_region.shape[0]
            # r = feature_region.shape[1]
            # 计算id损失
            # loss_id = torch.zeros(m).cuda()
            # for i_batch in range(m):
            #     V = torch.arange(v_min[i_batch], v_max[i_batch] + 1)
            #     s = class_score.permute(0, 2, 1)[:, :, V]
            #     t = target.view(m, 1).expand(m, r)[:, V]
            #     loss_id[i_batch] = F.cross_entropy(s, t)
            # l_id = torch.mean(loss_id)

            l_r = F.cross_entropy(probability_map, region_label)
            l_tri = triplet(feature_region, target, region_label)
            return l_r + l_tri[0], l_r,  l_tri[0]
    return loss_func_vpm_val


def make_loss(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'cluster':
        cluster = ClusterLoss(cfg.SOLVER.CLUSTER_MARGIN, True, True, cfg.SOLVER.IMS_PER_BATCH // cfg.DATALOADER.NUM_INSTANCE, cfg.DATALOADER.NUM_INSTANCE)
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_cluster':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        cluster = ClusterLoss(cfg.SOLVER.CLUSTER_MARGIN, True, True, cfg.SOLVER.IMS_PER_BATCH // cfg.DATALOADER.NUM_INSTANCE, cfg.DATALOADER.NUM_INSTANCE)
    else:
        print('expected METRIC_LOSS_TYPE should be triplet, cluster, triplet_cluster'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)
    elif cfg.DATALOADER.SAMPLER == 'triplet':
        def loss_func(score, feat, target):
            return triplet(feat, target)[0]
    elif cfg.DATALOADER.SAMPLER == 'softmax_triplet':
        def loss_func(score, feat, target):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return xent(score, target) + triplet(feat, target)[0]  # new add by luo, open label smooth
                else:
                    return F.cross_entropy(score, target) + triplet(feat, target)[0]    # new add by luo, no label smooth

            elif cfg.MODEL.METRIC_LOSS_TYPE == 'cluster':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return xent(score, target) + cluster(feat, target)[0]  # new add by luo, open label smooth
                else:
                    return F.cross_entropy(score, target) + cluster(feat, target)[0]    # new add by luo, no label smooth

            elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_cluster':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return xent(score, target) + triplet(feat, target)[0] + cluster(feat, target)[0]  # new add by luo, open label smooth
                else:
                    return F.cross_entropy(score, target) + triplet(feat, target)[0] + cluster(feat, target)[0]    # new add by luo, no label smooth
            else:
                print('expected METRIC_LOSS_TYPE should be triplet, cluster, triplet_cluster，'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    else:
        print('expected sampler should be softmax, triplet or softmax_triplet, '
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func


