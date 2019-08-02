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

        #当type为rnn时，需要用到self.rnn
        self.rnn = torch.nn.LSTM(input_size=512, hidden_size=512, num_layers=2, batch_first=True)

        #当type为stac时，需要用到
        self.stack_cnn1 = torch.nn.Conv1d(num_view, 10, stride=1, padding=0, kernel_size=1)
        self.stack_cnn2 = torch.nn.Conv1d(10, 1, stride=1, padding=0, kernel_size=1)


        # 将融合特征映射至新的空间
        self.fc1_map_ff =   torch.nn.Linear(in_features=self.length_fused_feature, out_features=self.length_fused_feature)
        self.fc2_map_ff = torch.nn.Linear(in_features=self.length_fused_feature, out_features=self.length_fused_feature)

        #将原始特征映射至新的空间
        self.fc1_map_f = torch.nn.Linear(in_features=self.length_fused_feature, out_features=self.length_fused_feature)
        self.fc2_map_f = torch.nn.Linear(in_features=self.length_fused_feature, out_features=self.length_fused_feature)



    # x为require feature , dimension = batch_size * feature_length
    # y为registered features  dimension = batch_size * num_views * feature_length
    def forward(self,ft):
        x = ft[:,0:-1,:]
        N, M, L = x.size()  # N为batch_size, M为Multi-view数量
        y=ft[:,-1,:]
        y = y.view(N,-1,L)
        output = []

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
            fused_feature = self.stack_cnn1(x)
            # print(fused_feature.shape)
            fused_feature = self.stack_cnn2(fused_feature)
            fused_feature = torch.squeeze(fused_feature,1)
            # print(fused_feature.shape)

        fused_feature = self.fc1_map_ff(fused_feature)
        fused_feature = torch.tanh(fused_feature)
        fused_feature = self.fc2_map_ff(fused_feature)

        y = self.fc1_map_f(y)
        y = torch.tanh(y)
        query_feature = self.fc2_map_ff(y)

        output.append(fused_feature)
        output.append(query_feature)
                
        if self.type=="unfused":
            result = result.view(N, M, -1)
            result = torch.max(result, 1)
        output.append(result)
        return output
