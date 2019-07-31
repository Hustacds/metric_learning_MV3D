# encoding: utf-8
"""
@author: Dong Shuai
@contact: dongshuai@zsc.edu.cn
"""

from torch.utils.data import Dataset
import torchvision.transforms as T

import numpy as np
import torch



class Feature_Dataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, feature_path):
        fe_data = np.load(feature_path)
        self.fe = fe_data[fe_data.files[0]]
        self.id = fe_data[fe_data.files[1]]
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.id)

    def get_id_nums(self):
        return np.max(self.id)

    def __getitem__(self, index):
        fe = self.fe[index]
        id = self.id[index][0]
        fe = torch.from_numpy(fe)
        return fe, id
