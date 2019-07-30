#coding=utf-8
#将ModelNet40数据集分割成实例分类和检索数据集
import os
import shutil

def main():
    orgpath = 'D:\科研\Paper\Fine-grained recognition of 3d objects based on multi-view recurrent networks\Dataset\shaded_images\modelnet40_images_new_12x'
    targetpath = 'D:\科研\Paper\Fine-grained recognition of 3d objects based on multi-view recurrent networks\Dataset\ModelNet40'

    objs = os.listdir(orgpath)
    print(objs)

    types = ['train','test']

    for objtype in objs:
        for type in types:
            views = os.listdir(orgpath + '\\'+ objtype + '\\' + type)
            print(views)
            for view in views:
                tmp =  view.split('.obj.')
                objname = tmp[0]
                viewname = tmp[0] +'_'+ tmp[1].split('_v')[1]
                print(objname,viewname)
                target_objpath = targetpath+ '\\' + type + '\\'+objname
                if  not os.path.exists(target_objpath):
                    # os.rmdir(target_objpath)
                    os.mkdir(target_objpath)
                shutil.copyfile(orgpath + '\\'+ objtype + '\\' + type + '\\' + view, target_objpath + '\\' + viewname)

if __name__ == '__main__':
    main()