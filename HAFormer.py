# 实现HAFormer的CNNdecoder
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.module import T


class CNNstem(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(CNNstem, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),  # [C,H/2,W/2]
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.conv(x)

class PEM(nn.Module):
    def __init__(self):
        super(PEM, self).__init__()
        self.GAP = nn.AvgPool2d(kernel_size=3,stride=1,padding=1) # 全局平均池化 [1*h*2]
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch ,_, h, w = x.size()  # 提取空间维度 :ml-citation{ref="1,2" data="citationList"}
        A = self.GAP(x)  # 输出形状 [batch, 1, h, w]
        A = A.reshape(batch,-1,1,1)  # 展平为 [batch, 1, h*w]
        A = self.softmax(A)  # 在 dim=2 上归一化 :ml-citation{ref="5,6" data="citationList"}
        A = A.reshape(batch, 1, h, w)  # 恢复为 [batch, 1, h, w]
        y=x*A  # 按元素相乘
        return self.relu(y+x)  #广播相加过后进行激活

# 通道混洗
class channelShuffle(nn.Module):
    def __init__(self,num_groups):
        super(channelShuffle, self).__init__()
        self.num_groups = num_groups
    def forward(self, x):
        batch,c,h,w = x.size()
        chn_per_group = c // self.num_groups
        x = x.view(batch,self.num_groups,chn_per_group,h,w)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch, -1, h, w)
        return x

class HAPE(nn.Module):
    def __init__(self,in_channels,dilation_rate):
        super(HAPE, self).__init__()
        self.reduce_channels = nn.Conv2d(in_channels, in_channels//4, kernel_size=1) # 通道将为四分之一
        #---------------非空洞卷积----------------------------
        self.conv1x3_1 = nn.Conv2d(in_channels//4, in_channels//4, kernel_size=(3,1), padding=(1,0))
        self.conv3x1_1 = nn.Conv2d(in_channels//4, in_channels//4, kernel_size=(1,3),padding=(0,1))
        #---------------空洞卷积------------------------------
        self.conv_3x1_2 = nn.Conv2d(in_channels//4, in_channels//4, kernel_size=(3, 1), dilation=dilation_rate,
                                    padding=(dilation_rate * (3 - 1) // 2, 0))
        self.conv_1x3_2 = nn.Conv2d(in_channels//4, in_channels//4, kernel_size=(1, 3), dilation=dilation_rate,
                                    padding=(0, dilation_rate * (3 - 1) // 2))

        self.conv_5x1 = nn.Conv2d(in_channels//4, in_channels//4, kernel_size=(5, 1), dilation=dilation_rate,
                                  padding=(dilation_rate * (5 - 1) // 2, 0))
        self.conv_1x5 = nn.Conv2d(in_channels//4, in_channels//4, kernel_size=(1, 5), dilation=dilation_rate,
                                  padding=(0, dilation_rate * (5 - 1) // 2))

        self.conv_7x1 = nn.Conv2d(in_channels//4, in_channels//4, kernel_size=(7, 1), dilation=dilation_rate,
                                  padding=(dilation_rate * (7 - 1) // 2, 0))
        self.conv_1x7 = nn.Conv2d(in_channels//4, in_channels//4, kernel_size=(1, 7), dilation=dilation_rate,
                                  padding=(0, dilation_rate * (7 - 1) // 2))
        self.pem = PEM()
        self.relu = nn.ReLU(inplace=True)
        self.add_channels = nn.Conv2d(in_channels //4 , in_channels, kernel_size=1)
        self.shuffle = channelShuffle(4)

    def forward(self, x):
        y = self.reduce_channels(x)
        y_1 = self.conv3x1_1(y)
        y_1 = self.conv1x3_1(y_1)
        y_2 = self.conv_3x1_2(y)
        y_2 = self.conv_1x3_2(y_2)
        y_3 = self.conv_5x1(y)
        y_3 = self.conv_1x5(y_3)
        y_4 = self.conv_7x1(y)
        y_4 = self.conv_1x7(y_4)
        pem_1 = self.pem(y_1)
        pem_2 = self.pem(y_2)
        pem_3 = self.pem(y_3)
        pem_4 = self.pem(y_4)
        pem = pem_1+pem_2+pem_3+pem_4
        X = self.add_channels(self.relu(pem))+x
        return self.shuffle(X)

# input_tensor = torch.randn(1, 4, 32, 32)
# model = HAPE(4, 3)
# output = model(input_tensor)
# print(output.shape)