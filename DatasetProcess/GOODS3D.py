#coding=utf-8
#将ModelNet40数据集分割成实例分类和检索数据集
import os
import shutil

rp = 'D:\科研\Paper\Fine-grained recognition of 3d objects based on multi-view recurrent networks\Dataset\GOODS3D\Test'
objs = os.listdir(rp)

for obj in objs:
    l = len(os.listdir(rp+'\\'+obj))
    if l<=100:
        print("物体{}的视图数量为{}".format(obj,l))