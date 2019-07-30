#coding=utf-8
#由于feature_exact_123模型训练数据已经丢失了，需要再统计其训练的准确率，我们是没有测试集的。
#在之前训练的模型基础上，继续训练


import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from tools.BasiceDataset import ImgDataset
import cv2
from models.Model import Model
import torch.optim as optim

#定义特征提取分类模型
class FEM(Model):
    def __init__(self, name, nclasses):
        super(FEM, self).__init__(name)
        self.net = models.resnet18(pretrained=True)
        self.fc = nn.Linear(512, nclasses)

    def forward(self, x):
        output = []
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        x = self.net.avgpool(x)
        x = x.view(x.size(0), -1)
        output.append(x)
        x = self.fc(x)
        output.append(x)
        return output


#              ModelNet40         MV3D         GOODS3D          ModelNet40+MV3D+GOODS3D
# train          200              65            109              200+65+109 = 374
# test           2468             30            40               2468+30+40 =  2538
#只用第4组进行训练，即使用ModelNet40+MV3D+GOODS3D，并且不划分测试集，即看在训练集上的准确率

print("初始化数据路径")
root_path = '/home/ds/MVRCNN/FGR3D/dataset'
train_dataset_list = [[root_path +'/ModelNet40/Train' ],[root_path +'/MV3D/Train'] ,[root_path +'/GOODS3D/Train'],[root_path +'/ModelNet40/Train' ,root_path +'/MV3D/Train' ,root_path +'/GOODS3D/Train']]
test_dataset_list = [[root_path +'/ModelNet40/Test' ],[root_path +'/MV3D/Test'] ,[root_path +'/GOODS3D/Test'],[root_path +'/ModelNet40/Test' ,root_path +'/MV3D/Test',root_path +'/GOODS3D/Test']]

run_mode = True    #true为训练， false为提取特征
LR = 0.001
train_batch_size = 64
test_batch_size = 1
train_shuffle=True
test_shuffle=False
EPOCHS = 50

dataset_index = 3

#fem训练用dataloader
train_dataset_path = train_dataset_list[dataset_index]
train_dataset =  ImgDataset(train_dataset_path,num_obj_threshold=200) #200,num_obj_threshold=10
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=train_shuffle, num_workers=0)

#抽取特征用loader
fe_train_dataset_path = train_dataset_list[dataset_index]
fe_train_dataset =  ImgDataset(fe_train_dataset_path,num_obj_threshold=2000)
fe_train_loader = torch.utils.data.DataLoader(fe_train_dataset, batch_size=1, shuffle=False, num_workers=0)

fe_test_dataset_path = test_dataset_list[dataset_index]
fe_test_dataset =  ImgDataset(fe_test_dataset_path,num_obj_threshold=2000)
fe_test_loader = torch.utils.data.DataLoader(fe_test_dataset, batch_size=1, shuffle=False, num_workers=0)
# fem = FEM('Feature_Exact_Model',nclasses = 374)
# for i, data in enumerate(train_loader):
#     fem.eval()
#     print(data[0].numpy())
#     label = data[0].numpy()[0]
#     img = data[1].numpy()[0]
#     print(label)
#
#     out = fem(data[1])
#     print( out[0].shape)
#     print(out[1].shape)
#     img = img.transpose((1,2,0))
#     img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
#     cv2.imshow('instance',img)
#     if cv2.waitKey(1) & 0xff == ord('q'):
#         break

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 模型定义-ResNet
net = FEM('Feature_Exact_Model',nclasses = 374).to(device)
net.load_state_dict(torch.load('/home/ds/MVRCNN/FGR3D/models/feature_exact_123/feature_exact_123.pth'))
# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题


# 训练
if __name__ == "__main__":
    if run_mode:
        best_acc = 85  #2 初始化best test accuracy
        print("Start Training, Resnet-18!")  # 定义遍历数据集的次数
        with open("/home/ds/MVRCNN/FGR3D/models/feature_exact_123/acc.txt", "w") as f:
            with open("/home/ds/MVRCNN/FGR3D/models/feature_exact_123/log.txt", "w")as f2:
                for epoch in range(20):
                    # if epoch<10:
                    #     LR = 0.001
                    # if epoch >=10 and epoch<20:
                    #     LR = 0.0001
                    # if epoch>=20 and epoch <30:
                    #     LR = 0.00001
                    # if epoch>=30 and epoch <40:
                    #     LR = 0.000001
                    # if epoch>=40:
                    #     LF = 0.0000001
                    #
                    # optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9,
                    #                       weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
                    #
                    # print('\nEpoch: %d' % (epoch + 1))
                    # net.train()
                    # sum_loss = 0.0
                    # correct = 0.0
                    # total = 0.0
                    # for i, data in enumerate(train_loader):
                    #     # 准备数据
                    #     length = len(train_loader)
                    #     inputs = data[1]
                    #     labels = data[0]
                    #     # inputs, labels = data
                    #     inputs, labels = inputs.to(device), labels.to(device)
                    #     optimizer.zero_grad()
                    #
                    #     # forward + backward
                    #     outputs_group = net(inputs)
                    #     features = outputs_group[0]
                    #     outputs = outputs_group[1]
                    #     loss = criterion(outputs, labels)
                    #     loss.backward()
                    #     optimizer.step()
                    #
                    #     # 每训练1个batch打印一次loss和准确率
                    #     sum_loss += loss.item()
                    #     _, predicted = torch.max(outputs.data, 1)
                    #     total += labels.size(0)
                    #     correct += predicted.eq(labels.data).cpu().sum()
                    #     print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                    #           % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    #     f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                    #           % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    #     f2.write('\n')
                    #     f2.flush()

                    # 每训练完一个epoch测试一下准确率
                    print("Waiting Test!")
                    with torch.no_grad():
                        correct = 0
                        total = 0
                        for i, data in enumerate(train_loader):
                        # for data in train_loader:
                            net.eval()
                            images = data[1]
                            labels = data[0]
                            images, labels = images.to(device), labels.to(device)
                            outputs_group = net(images)
                            outputs = outputs_group[1]
                            # 取得分最高的那个类 (outputs.data的索引号)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum()
                        print('测试分类准确率为：%.3f%%' % (100 * correct / total))
                        acc = 100. * correct / total
                        print(correct)
                        print(total)
                        # 将每次测试结果实时写入acc.txt文件中
                        print('Saving model......')
                        torch.save(net.state_dict(), '%s/net_%03d.pth' % ('/home/ds/MVRCNN/FGR3D/models/feature_exact_123', epoch + 1))
                        f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                        f.write('\n')
                        f.flush()
                        # 记录最佳测试分类准确率并写入best_acc.txt文件中
                        if acc > best_acc:
                            f3 = open("best_acc.txt", "w")
                            f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                            f3.close()
                            best_acc = acc
                print("Training Finished, TotalEPOCH=%d" % EPOCHS)
    else:
        with torch.no_grad():
            net.load_state_dict(torch.load('/home/ds/MVRCNN/FGR3D/models/feature_exact_123'+'/net_010.pth'))
            net.eval()
            fe_train = []
            labels_train = []
            for i, data in enumerate(fe_train_loader):
                print(i)
                images = data[1]
                labels = data[0]
                images, labels = images.to(device), labels.to(device)
                outputs_group = net(images)
                features = outputs_group[0]
                labels = labels.cpu().numpy()
                features = features.cpu().numpy()[0]
                # print(labels)
                # print(features)
                fe_train.append(features)
                labels_train.append(labels)
            np.savez("/home/ds/MVRCNN/FGR3D/dataset/fe_"+str(dataset_index+1)+"/fe_train.npz",fe_train =fe_train,labels_train = labels_train )


            fe_test = []
            labels_test = []
            for i, data in enumerate(fe_test_loader):
                print(i)
                images = data[1]
                labels = data[0]
                images, labels = images.to(device), labels.to(device)
                outputs_group = net(images)
                features = outputs_group[0]
                labels = labels.cpu().numpy()
                features = features.cpu().numpy()[0]
                fe_test.append(features)
                labels_test.append(labels)
            np.savez("/home/ds/MVRCNN/FGR3D/dataset/fe_"+str(dataset_index+1)+"/fe_test.npz", fe_test=fe_test, labels_test=labels_test)