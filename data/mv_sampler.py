# encoding: utf-8
"""
@author: Dong Shuai
@contact: dongshuai@zsc.edu.cn
"""
import copy
import random
import torch
from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import Sampler

# 方法和MVRNN使用的思路是一样的。这样可以实现不同的组合
class RandomMVSampler(Sampler):
    """
    随机抽取
    """
    def __init__(self, data_source, batch_size,num_view_to_fuse,length_per_id):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_view_sample = num_view_to_fuse+1
        self.length_per_id = length_per_id

        self.index_dic = defaultdict(list)
        for index, (_, id) in enumerate(data_source):
            self.index_dic[id].append(index)
        self.ids = list(self.index_dic.keys())
        self.num_identities = len(self.ids)  #样本物体的总数

    def __iter__(self):
        ret = []
        for j in range(self.length_per_id):
            indices = torch.randperm(self.num_identities)
            for i in indices:
                id = self.ids[i]
                t = self.index_dic[id]
                replace = False if len(t) >= self.num_view_sample else True
                t = np.random.choice(t, size=self.num_view_sample, replace=replace)
                ret.extend(t)
        return iter(ret)

    def __len__(self):
        return self.length_per_id * self.num_identities * self.num_view_sample