from tools.BasiceDataset import ImgDataset
import torch
import cv2

import numpy as np

# print("初始化数据路径")
# root_path = 'D:\科研\Paper\FGR3D\dataset'
# train_dataset_list = [[root_path +'\ModelNet40\Train' ],[root_path +'\MV3D\Train'] ,[root_path +'\GOODS3D\Train'],[root_path +'\ModelNet40\Train' ,root_path +'\MV3D\Train' ,root_path +'\GOODS3D\Train']]
# test_dataset_list = [[root_path +'\ModelNet40\Test' ],[root_path +'\MV3D\Test'] ,[root_path +'\GOODS3D\Test'],[root_path +'\ModelNet40\Test' ,root_path +'\MV3D\Test',root_path +'\GOODS3D\Test']]
#
# train_batch_size = 64
# test_batch_size = 1
# train_shuffle=True
# test_shuffle=False
#
#
# train_loader_list = []
# test_loader_list = []
#
# for i in range(4):
#     train_dataset_path = train_dataset_list[i]
#     train_dataset =  ImgDataset(train_dataset_path,num_obj_threshold=200)
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=train_shuffle, num_workers=0)
#     train_loader_list.append(train_loader)
#
#     test_dataset_path = test_dataset_list[i]
#     test_dataset = ImgDataset(test_dataset_path)
#     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=test_shuffle,num_workers=0)
#     test_loader_list.append(test_loader)

