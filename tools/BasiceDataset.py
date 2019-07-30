#coding=utf-8
#将ModelNet40数据集分割成实例分类和检索数据集
import numpy as np
import glob
import torch.utils.data
import os
import math
from skimage import io, transform
from PIL import Image
import torch
import torchvision as vision
from torchvision import transforms, datasets
import random

# 基本的Dataset，对于不同的数据集，采用不同
# 三个数据集结构类似，所以应该可以通用
# 在特征提取网络的训练阶段，dataset需要提取不同物体不同类别，
# 4个特征提取网络，前个针对单个数据集，第4个对三个数据集进行整合，
# modelnet物体数量较大，直接做分类是否合理？需要提取部分物体进行识别吗？
# 三个数据集组合的时候，ModelNet取200个物体应该比较合理。
#这个类如果要设计成可以融合

#单个数据集时，label对应的就是计数
#多个数据集时，modelnet的视图数量比较少，是否会造成影响呢？
#该dataset只需要解决通用特征提取网络的训练即可

class ImgDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir,num_obj_threshold =20000000):
        self.num_dataset = len(root_dir)
        self.root_dir = root_dir
        print("融合{}个数据集".format(self.num_dataset))
        self.obj_list = []
        self.classnames = []

        for i in range(self.num_dataset):
            objpath = self.root_dir[i]
            objlist = os.listdir(objpath)
            num_obj = len(objlist)
            print("在路径{}下，共找到实例{}个，其路径为{}".format(objpath,num_obj,objlist))
            if num_obj >=num_obj_threshold:
                num_obj = num_obj_threshold
                print("样本数量超过200个，因此只取前200个")
            for j in range(num_obj):
                obj = objlist[j]
                self.obj_list.append(objpath + '/'+obj)
                self.classnames.append(obj)
        # print(self.obj_list)
        # print(self.classnames)
        # print("共有物体实例{}个".format(len(self.obj_list)))

        self.sample_list = []
        self.label_list =[]
        for k in range(len(self.obj_list)):
            view_path = self.obj_list[k]
            viewlist = os.listdir(view_path)
            for view in viewlist:
                self.sample_list.append(view_path+"/" + view)
                self.label_list.append(k)

        print(len(self.sample_list))
        # for i in range(len(self.sample_list)):
        #     print("图片路径为{}，标签为{}".format(self.sample_list[i],self.label_list[i]))

        self.transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])

        ])


    def __len__(self):
        return len(self.sample_list)


    def __getitem__(self, idx):
        # print("dataset获取{}样本,标签为{}".format(idx,self.label_list[idx]))
        path = self.sample_list[idx]
        class_id = self.label_list[idx]
        # Use PIL instead
        im = Image.open(path).convert('RGB')
        im = im.resize((224,224))
        if self.transform:
            im = self.transform(im)
        # print(im.shape)
        return (class_id, im, path)

