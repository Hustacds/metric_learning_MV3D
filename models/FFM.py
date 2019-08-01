# encoding: utf-8
"""
@author: Dong Shuai
@contact: dongshuai@zsc.edu.cn
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
#特征聚合网络的定义
class FFM(nn.Module):
    def __init__(self,net_type,loss_type,num_view,class_num):
        super(FFM,self).__init__()
        self.net_type = net_type
        self.loss_type = loss_type
        self.num_view = num_view
        self.class_num = class_num
        if self.net_type =="concatenating":
            self.length_fused_feature = 512 * (num_view)
        else:
            self.length_fused_feature = 512

        self.rnn = torch.nn.LSTM(input_size=512, hidden_size=512, num_layers=2, batch_first=True)
        self.classify = torch.nn.Linear(in_features=self.length_fused_feature, out_features=2)
        self.classify_middle1 = torch.nn.Linear(in_features=self.length_fused_feature, out_features=self.length_fused_feature)
        self.classify_middle2 = torch.nn.Linear(in_features=self.length_fused_feature,
                                             out_features=self.length_fused_feature)


        self.classify_cnn1 = torch.nn.Conv1d(num_view,10,stride=1,padding = 0,kernel_size=1)
        self.classify_cnn2 = torch.nn.Conv1d(10, 1, stride=1, padding=0, kernel_size=1)

    # x为require feature , dimension = batch_size * feature_length
    # y为registered features  dimension = batch_size * num_views * feature_length
    def forward(self,x):
        output = []
        N, M, L = x.size()  # N为batch_size, M为Multi-view数量
        if self.type=="unfused":
            # print('batchsize = {}, 视图数量为 {}, 检索特征长度 = {}'.format(N,M, L))
            fused_feature = x.view(N*M, L)

        elif self.type=='rnn':
            # print('batchsize = {}, 视图数量为 {}, 检索特征长度 = {}'.format(N, M, L))
            h0 = Variable(torch.zeros(2, x.size(0), 512).cuda())
            c0 = Variable(torch.zeros(2, x.size(0), 512).cuda())
            out,(h_n,c_n) = self.rnn(x, (h0, c0))
            fused_feature = out[:,-1,:]

        elif self.type =='maxpooling':
            # print('batchsize = {}, 检索特征长度 = {}'.format(N, L))
            fused_feature = torch.max(x, 1)[0]

        elif self.type =="averaging":
            # print('batchsize = {}, 视图数量为 {}, 检索特征长度 = {}'.format(N, M, L))
            fused_feature = torch.mean(x, 1)

        elif self.type =="concatenating":
            # print('batchsize = {}, 视图数量为 {}, 检索特征长度 = {}'.format(N, M, L))
            fused_feature = x.view(N, M * L)

        elif self.type == 'stack':
            # print('batchsize = {}, 视图数量为 {}, 检索特征长度 = {}'.format(N, M, L))
            fused_feature = self.classify_cnn1(x)
            # print(fused_feature.shape)
            fused_feature = self.classify_cnn2(fused_feature)
            fused_feature = torch.squeeze(fused_feature,1)
            # print(fused_feature.shape)

        output.append(fused_feature)
        input = self.classify_middle1(fused_feature)
        input = torch.tanh(input)
        input = self.classify_middle2(input)
        input = torch.tanh(input)
        if self.loss_type=='cosine':
            input = nn.BatchNorm1d(input)
        result = self.classify(input)  # N *
        if self.type=="unfused":
            result = result.view(N, M, -1)
            result = torch.max(result, 1)
        output.append(result)
        return output
