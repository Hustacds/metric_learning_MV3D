# encoding: utf-8
"""
@author:  Dong Shuai
@contact: dong@gmail.com
"""
# from .cuhk03 import CUHK03
from .dataset_loader import ImageDataset

__factory = {
    # 'market1501': Market1501,
    # 'cuhk03': CUHK03,
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)
