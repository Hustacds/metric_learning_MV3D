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
        self.length_feature = 512
        self.net_type = net_type
        self.loss_type = loss_type
        self.num_view = num_view
        self.class_num = class_num
        if self.net_type =="concatenating":
            self.length_fused_feature = self.length_feature * (num_view)
        else:
            self.length_fused_feature = self.length_feature

        #当type为rnn时，需要用到self.rnn
        self.rnn = torch.nn.LSTM(input_size=self.length_feature, hidden_size=self.length_feature, num_layers=2, batch_first=True)

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
        N,L = ft.size()  # N为batch_size，L为特征长度
        N = int(N/(self.num_view+1))
        ft = ft.view(self.num_view+1,N,L)
        ft = torch.transpose(ft,0,1)

        x = ft[:,0:-1,:]
        N, M, L = x.size()  # N为batch_size, M为Multi-view数量
        y=ft[:,-1,:]
        output = []

        if self.net_type=="unfused":
            # print('batchsize = {}, 视图数量为 {}, 检索特征长度 = {}'.format(N,M, L))
            fused_feature = x.contiguous().view(N*M, L)

        elif self.net_type=='rnn':
            # print('batchsize = {}, 视图数量为 {}, 检索特征长度 = {}'.format(N, M, L))
            h0 = Variable(torch.zeros(2, x.size(0), self.length_feature).cuda())
            c0 = Variable(torch.zeros(2, x.size(0), self.length_feature).cuda())
            out,(h_n,c_n) = self.rnn(x, (h0, c0))
            fused_feature = out[:,-1,:]

        elif self.net_type =='maxpooling':
            # print('batchsize = {}, 检索特征长度 = {}'.format(N, L))
            fused_feature = torch.max(x, 1)[0]

        elif self.net_type =="averaging":
            # print('batchsize = {}, 视图数量为 {}, 检索特征长度 = {}'.format(N, M, L))
            fused_feature = torch.mean(x, 1)

        elif self.net_type =="concatenating":
            # print('batchsize = {}, 视图数量为 {}, 检索特征长度 = {}'.format(N, M, L))
            fused_feature = x.view(N, M * L)

        elif self.net_type == 'stack':
            # print('batchsize = {}, 视图数量为 {}, 检索特征长度 = {}'.format(N, M, L))
            fused_feature = self.stack_cnn1(x)
            # print(fused_feature.shape)
            fused_feature = self.stack_cnn2(fused_feature)
            fused_feature = torch.squeeze(fused_feature,1)
            # print(fused_feature.shape)

        fused_feature = self.fc1_map_ff(fused_feature)
        fused_feature = torch.tanh(fused_feature)
        fused_feature = self.fc2_map_ff(fused_feature)

        #这个问题的本质是融合特征和单视图特征能否映射到同一个空间进行度量
        #把fused特征映射到新的空间
        fused_feature = fused_feature.view(N, -1, self.length_feature)
        output.append(fused_feature)   # N * W * L   N为batchsize, W暂时定义为feature宽度吧，L为feature长度

        #把query特征映射至新的空间
        y = self.fc1_map_f(y)
        y = torch.tanh(y)
        y = self.fc2_map_f(y)
        output.append(y)
        return output
