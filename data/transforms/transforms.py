# encoding: utf-8

import math
import random
import numpy as np
import torchvision.transforms as T
import torch
import cv2

def cv2_imshow(window_title,img):
    img_np = img.cpu().numpy()
    img_np = cv2.cvtColor(img_np,cv2.COLOR_BGR2RGB)
    cv2.imshow(window_title,img_np)

class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

# crop_pos: 裁剪的方位，上、下，上下同时
# crop_ratio 裁剪的最大比例限制，0,5
# m_div 子区域划分的行数
# n_div 子区域划分的列数
class RandomCrop(object):
    def __init__(self,crop_pos,crop_ratio,m_div,n_div,input_size):
        self.crop_pos = crop_pos
        self.crop_ratio = crop_ratio
        self.m_div = m_div
        self.n_div = n_div
        self.input_size = input_size

        #计算256*128的原图上的区域划分节点
        self.m_point = np.linspace(0,self.input_size[0],self.m_div+1)
        self.n_point = np.linspace(0, self.input_size[1], self.n_div + 1)

        self.to_tensor = T.ToTensor()
        self.to_pil = T.ToPILImage()
        self.resize = T.Resize([256,128])
        self.range_points = []
        for i in range(m_div):
            for j in range(n_div):
                self.range_points.append([self.m_point[i],self.n_point[j],self.m_point[i+1],self.n_point[j+1]])

        self.count = 0

    def __call__(self, img_holistic):
        # print("transform--RandomCrop:img_holistic.shape = {}".format(img_holistic.size))
        # channels = img_holistic.shape[0]
        h = img_holistic.size[1]
        w = img_holistic.size[0]
        # print("transform--RandomCrop:channels = {}, h={}, w={}".format(channels,h,w))
        random_crop_ratio = self.crop_ratio * random.random()
        # random_crop_ratio = self.crop_ratio
        # crop_ratio = 0.5
        # print(crop_ratio)
        if self.crop_pos == 'bottom':
            w_min =0
            w_max = w
            h_min = 0
            h_max = round(h * (1 - random_crop_ratio))
        elif self.crop_pos == 'top':
            w_min = 0
            w_max = w
            h_min = round(h * random_crop_ratio)
            h_max = h
        elif self.crop_pos == 'bilateral':
            w_min = 0
            w_max = w
            h_min = round(h * random_crop_ratio / 2)
            h_max = round(h * (1 - random_crop_ratio / 2))
        else:
            print('wrong crop position setting')

        # 子区域划分节点在图上的坐标
        point_feature = []

        #计算各区域在特征图上的坐标
        for p in self.range_points:
            x_lt = (p[0]-h_min)*8/(h_max-h_min)
            y_lt= (p[1] - w_min) *4 /(w_max - w_min)
            x_rb = (p[2] - h_min) * 8 / (h_max - h_min)
            y_rb = (p[3] - w_min) * 4 / (w_max - w_min)
            point_feature.append([x_lt,y_lt,x_rb,y_rb])


        label_region  = torch.zeros([8,4]).long()-1
        #区域从0开始，先行后列
        for i in range(8):
            for j in range(4):
                for k in range(len(point_feature)):
                    pf = point_feature[k]
                    if pf[0]<=i and pf[2]>i and pf[1]<=j and pf[3]>j:
                        label_region[i,j] = k
        # print(point_feature)
        # print(label_region)

        # img_p = self.to_tensor(img_holistic)
        # img_p = img_p.permute(1, 2, 0)
        # img_p = img_p.cpu().numpy()
        # img_p = cv2.cvtColor(img_p,cv2.COLOR_RGB2BGR)
        # for p in self.range_points:
        #     img_p = cv2.rectangle(img_p,(int(p[1]),int(p[0])),(int(p[3]),int(p[2])), (0, 255, 0),1)

        # img_p = cv2.rectangle(img_p,(int(w_min),int(h_min)),(int(w_max),int(h_max)), (255, 0, 0),2)

        # cv2.imshow('original img', img_p)
        # cv2.waitKey(1000)

        img_partial = img_holistic.crop([w_min,h_min,w_max,h_max])

        # img_region = np.zeros((256,128,3),np.float32)
        # for i in range(256):
        #     for j in range(128):
        #         if label_region[i//32,j//32] == 0 :
        #             img_region[i,j,:] = [0,0,0]
        #         if label_region[i//32,j//32] == 1 :
        #             img_region[i,j,:] = [0,0,1]
        #         if label_region[i//32,j//32] == 2:
        #             img_region[i, j, :] = [0, 1, 1]
        #         if label_region[i//32,j//32] == 3:
        #             img_region[i, j, :] = [0, 1, 0]
        #         if label_region[i//32,j//32] == 4:
        #                 img_region[i, j, :] = [1, 1, 1]
        #         if label_region[i//32,j//32] == 5:
        #                 img_region[i, j, :] = [1, 0, 1]
        #         if label_region[i//32,j//32] == 6:
        #             img_region[i, j, :] = [1, 1, 0]
        #         if label_region[i//32,j//32] == 7:
        #             img_region[i, j, :] = [1, 0, 0]
        # img_region = cv2.resize(img_region,(128,256),cv2.INTER_NEAREST)
        #
        # img_p = self.to_tensor(img_partial)
        # img_p = img_p.permute(1, 2, 0)
        # img_p = img_p.cpu().numpy()
        # img_p = cv2.cvtColor(img_p, cv2.COLOR_RGB2BGR)
        # img_p = cv2.resize(img_p,(128,256))
        # img_add = cv2.addWeighted(img_p, 0.6, img_region, 0.4, 0)
        # cv2.imshow('partical img', img_add)
        # cv2.imwrite('D:\\Yolo\\PartialReID\\imgsamples\\'+str(self.count)+'_3_1.jpg', np.asarray(img_add*255, np.uint8))
        # self.count +=1
        # img_partial = self.to_pil(img_partial)
        # print('crop 以后的图像尺寸{}'.format(img_partial.size))
        # img_partial = self.resize(img_partial)
        # print('resize 以后的图像尺寸{}'.format(img_partial.size))
        # img_partial = self.to_tensor(img_partial)
        # print('to tensor 以后的图像尺寸{}'.format(img_partial.shape))


        #
        # img_p = img_partial.permute(0,2,3,1)
        # for i in range(batch_size):
        #     img_s = img_p[i,:]
        #     cv2_imshow('croped img',img_s)
        #     if cv2.waitKey(1000) & 0xFF == ord('q'):
        #         break

        # batch_size * channel * height * width
        return img_partial,label_region


