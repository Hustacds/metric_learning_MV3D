# encoding: utf-8
"""
@author: Dong Shuai
@contact: dongshuai@zsc.edu.cn
"""
import numpy as np
from data.feature_dataset import Feature_Dataset
from torch.utils.data import DataLoader
from data.collate_batch import *
from data.mv_sampler import RandomMVSampler

def make_data_loader(cfg,feature_group,view_num):
    train_feature_path = cfg.DATASETS.ROOT_DIR + feature_group +  '\\fe_train.npz'
    train_set = Feature_Dataset(train_feature_path)
    train_id_nums = train_set.get_id_nums()+1
    train_loader = DataLoader(train_set,batch_size=cfg.SOLVER.OBJS_PER_BATCH*(view_num+1),
                              sampler=RandomMVSampler(train_set, cfg.SOLVER.OBJS_PER_BATCH, view_num,cfg.SOLVER.SAMPLER_LENGTH),
                              collate_fn=train_collate_fn)

    val_feature_path = cfg.DATASETS.ROOT_DIR + feature_group +  '\\fe_test.npz'
    val_set = Feature_Dataset(val_feature_path)
    val_id_nums = val_set.get_id_nums()+1
    val_loader = DataLoader(val_set,batch_size=cfg.TEST.OBJS_PER_BATCH*(view_num+1),
                              sampler=RandomMVSampler(val_set, cfg.TEST.OBJS_PER_BATCH*(view_num+1), view_num,cfg.TEST.SAMPLER_LENGTH),
                              collate_fn=train_collate_fn)
    return train_loader,val_loader,train_id_nums,val_id_nums