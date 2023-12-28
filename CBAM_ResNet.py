#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : CBAM_ResNet.py
@Author  : cao xu
@Time    : 2023/12/28 9:13
"""

import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class channel_Weight(nn.Module):
    def __init__(self,inchannel,ratio=16,pool_type=["avg","max"]):
        super(channel_Weight, self).__init__()
        self.fc=nn.Sequential(Flatten(),
                              nn.Linear(inchannel,inchannel//ratio,bias=False),
                              nn.ReLU(inplace=True),
                              nn.Linear(inchannel//ratio,inchannel,bias=False))
        self.pool=pool_type
    def forward(self,x):
        sum=None
        for i in self.pool:
            if i=="avg":
                avg=F.avg_pool2d(x,(x.size(2),x.size(3)),stride=(x.size(2),x.size(3)))
                #C*H*W---->1*H*W
                feature=self.fc(avg)
            elif i=="max":
                max=F.max_pool2d(x,(x.size(2),x.size(3)),stride=(x.size(2),x.size(3)))
                feature=self.fc(max)
            if sum is None:
                sum=feature
            else:
                sum+=feature

        weight=F.sigmoid(sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return weight*x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class  Spatial_weight(nn.Module):
    def __init__(self):
        super(Spatial_weight, self).__init__()
        self.pool=ChannelPool()
        self.conv=nn.Sequential(nn.Conv2d(in_channels=2,out_channels=1,kernel_size=7,stride=1,padding=3),
                                nn.BatchNorm2d(1,eps=1e-5, momentum=0.01, affine=True))

    def forward(self,x):
        spatial=self.pool(x)
        weight=self.conv(spatial)
        weight=F.sigmoid(weight)
        return x*weight


class CBAM(nn.Module):
    def __init__(self,inchannel,ratio=16,pool_type=["avg","max"]):
        super(CBAM, self).__init__()
        self.channnel_Weight=channel_Weight(inchannel,ratio=ratio,pool_type=pool_type)
        self.Spatial_weight=Spatial_weight()

    def forward(self,x):
        x=self.channnel_Weight(x)
        x=self.Spatial_weight(x)
        return x


class block(nn.Module):
    def __init__(self,inchannels,outchannels,stride=1,shortcut=False):
        super(block, self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels=inchannels,out_channels=outchannels//2,kernel_size=1,stride=stride),
            nn.BatchNorm2d(outchannels//2),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=outchannels//2, out_channels=outchannels // 2, kernel_size=3,padding=1,groups=32,stride=1),
            nn.BatchNorm2d(outchannels // 2),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=outchannels // 2, out_channels=outchannels, kernel_size=1,stride=1),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True)
        )
        self.layer4=CBAM(outchannels)
        self.shortcut=shortcut
        if shortcut:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channels=inchannels,out_channels=outchannels,kernel_size=1,stride=stride),
                nn.BatchNorm2d(outchannels)
            )

    def forward(self,x):
        if self.shortcut:
            resduial=self.shortcut(x)
        else:
            resduial=x
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x=x+resduial
        return x


class CBAM_Resnext(nn.Module):
    def __init__(self,layer_number,num_class):
        super(CBAM_Resnext, self).__init__()
        if layer_number==50:
            self.layer=[3,4,6,3]
        elif layer_number==101:
            self.layer=[3,4,23,3]
        elif layer_number==152:
            self.layer=[3,8,36,3]
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,padding=1,stride=2)
        )
        self.layer2=self.make_layer(64,256,1,self.layer[0])
        self.layer3 = self.make_layer(256, 512, 2, self.layer[1])
        self.layer4 = self.make_layer(512, 1024, 2, self.layer[2])
        self.layer5 = self.make_layer(1024, 2048, 2, self.layer[2])
        self.layer6=nn.AvgPool2d(kernel_size=7,stride=1)
        self.layer7=nn.Linear(2048,num_class)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def make_layer(self, inchannels, outchannels, stride, num):
        layers=[]
        layers.append(block(inchannels, outchannels, stride=stride, shortcut=True))
        for i in range(1,num):
            layers.append(block(outchannels, outchannels, stride=1, shortcut=False))
        return nn.Sequential(*layers)

    def forward(self,x):
        x=self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x=torch.flatten(x,1)
        print(x.size())
        x=self.layer7(x)
        return F.softmax(x,dim=1)


if __name__ == '__main__':
    device = torch.device("cuda:0")
    Xception = CBAM_Resnext(50, 2).to(device)
    # from torchsummary import summary
    # summary(Xception,(3,224,224))
