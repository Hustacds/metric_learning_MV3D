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
    def __init__(self,type,num_view):
        super(FFM,self).__init__()
        self.type = type
        self.num_view = num_view
        if self.type == 'rnn':
            self.length_fused_feature = 1024
        elif self.type == 'maxpooling':
            self.length_fused_feature = 1024
        elif self.type == "averaging":
            self.length_fused_feature = 1024
        elif self.type == "concatenating":
            self.length_fused_feature = 512*(num_view+1)
        elif self.type == "unfused":
            self.length_fused_feature = 1024
        elif self.type =='stack':
            self.length_fused_feature = 1024
        else:
            print("================wrong type===================")
        self.rnn = torch.nn.LSTM(input_size=512, hidden_size=512, num_layers=2, batch_first=True)
        self.match = torch.nn.Linear(in_features=self.length_fused_feature, out_features=2)
        self.match_middle1 = torch.nn.Linear(in_features=self.length_fused_feature, out_features=self.length_fused_feature)
        self.match_middle2 = torch.nn.Linear(in_features=self.length_fused_feature,
                                             out_features=self.length_fused_feature)


        self.match_cnn1 = torch.nn.Conv1d(num_view,10,stride=1,padding = 0,kernel_size=1)
        self.match_cnn2 = torch.nn.Conv1d(10, 1, stride=1, padding=0, kernel_size=1)

    # x为require feature , dimension = batch_size * feature_length
    # y为registered features  dimension = batch_size * num_views * feature_length
    def forward(self,x,y):

        if self.type=='rnn':
            N, L = x.size()  # N为batch_size, L为特征长度
            # print('batchsize = {}, 检索特征长度 = {}'.format(N, L))
            N, M, L = y.size()  # N为batch_size, M为Multi-view数量
            # print('batchsize = {}, 视图数量为 {}, 检索特征长度 = {}'.format(N, M, L))
            output = []
            h0 = Variable(torch.zeros(2, y.size(0), 512).cuda())
            c0 = Variable(torch.zeros(2, y.size(0), 512).cuda())
            out,(h_n,c_n) = self.rnn(y, (h0, c0))
            fused_feature = out[:,-1,:]
            output.append(fused_feature)
            input = torch.cat((x, fused_feature), 1)
            input = self.match_middle1(input)
            input = torch.tanh(input)
            input = self.match_middle2(input)
            input = torch.tanh(input)
            result = self.match(input) # N * 2
            output.append(result)
            return output

        elif self.type =='maxpooling':
            N, L = x.size()  # N为batch_size, L为特征长度
            # print('batchsize = {}, 检索特征长度 = {}'.format(N, L))
            N, M, L = y.size()  # N为batch_size, M为Multi-view数量
            # print('batchsize = {}, 视图数量为 {}, 检索特征长度 = {}'.format(N, M, L))
            output = []
            fused_feature = torch.max(y, 1)[0]
            output.append(fused_feature)
            # print(x.shape)
            # print(fused_feature.shape)
            input = torch.cat((x, fused_feature),1)
            # print(input.shape)
            input = self.match_middle1(input)
            input = torch.tanh(input)
            input = self.match_middle2(input)
            input = torch.tanh(input)
            result = self.match(input)  # N * 2
            # print(result.shape)
            output.append(result)
            return output

        elif self.type =="averaging":
            N, L = x.size()  # N为batch_size, L为特征长度
            # print('batchsize = {}, 检索特征长度 = {}'.format(N, L))
            N, M, L = y.size()  # N为batch_size, M为Multi-view数量
            # print('batchsize = {}, 视图数量为 {}, 检索特征长度 = {}'.format(N, M, L))
            output = []
            fused_feature = torch.mean(y, 1)
            output.append(fused_feature)
            input = torch.cat((x, fused_feature), 1)
            input = self.match_middle1(input)
            input = torch.tanh(input)
            input = self.match_middle2(input)
            input = torch.tanh(input)
            result = self.match(input)  # N * 2
            output.append(result)
            return output

        elif self.type =="concatenating":
            N, L = x.size()  # N为batch_size, L为特征长度
            # print('batchsize = {}, 检索特征长度 = {}'.format(N, L))
            N, M, L = y.size()  # N为batch_size, M为Multi-view数量
            # print('batchsize = {}, 视图数量为 {}, 检索特征长度 = {}'.format(N, M, L))
            output = []
            fused_feature = y.view(N, M * L)
            output.append(fused_feature)
            input = torch.cat((x, fused_feature), 1)
            input = self.match_middle1(input)
            input = torch.tanh(input)
            input = self.match_middle2(input)
            input = torch.tanh(input)
            result = self.match(input)
            output.append(result)
            return output  # N * 2

        elif self.type=="unfused":
            N, L = x.size()    #N为batch_size, L为特征长度
            # print('batchsize = {}, 检索特征长度 = {}'.format(N,L))
            N, M, L = y.size()  #N为batch_size, M为Multi-view数量
            # print('batchsize = {}, 视图数量为 {}, 检索特征长度 = {}'.format(N,M, L))
            output = []
            output.append(y)
            result = torch.Tensor(N,M,2).cuda()           # N * M * 2
            for i in range(M):
                input = torch.cat((x,y[:,i,:]),1)
                # print(input.shape)                # N * 2L
                input = self.match_middle1(input)
                input = torch.tanh(input)
                input = self.match_middle2(input)
                input = torch.tanh(input)
                result[:,i,:] = self.match(input)

            # print(result.shape)
            result = torch.mean(result,1)
            # print(result.shape)
            output.append(result)
            return output          # N * 2

        elif self.type == 'stack':
            N, L = x.size()  # N为batch_size, L为特征长度
            # print('batchsize = {}, 检索特征长度 = {}'.format(N, L))
            N, M, L = y.size()  # N为batch_size, M为Multi-view数量
            # print('batchsize = {}, 视图数量为 {}, 检索特征长度 = {}'.format(N, M, L))
            output = []
            fused_feature = self.match_cnn1(y)
            # print(fused_feature.shape)
            fused_feature = self.match_cnn2(fused_feature)
            fused_feature = torch.squeeze(fused_feature,1)
            # print(fused_feature.shape)
            output.append(fused_feature)
            input = torch.cat((x, fused_feature), 1)
            input = self.match_middle1(input)
            input = torch.tanh(input)
            input = self.match_middle2(input)
            input = torch.tanh(input)
            result = self.match(input)
            output.append(result)
            return output  # N * 2

