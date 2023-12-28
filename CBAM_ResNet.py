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
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet18_cbam', 'resnet34_cbam', 'resnet50_cbam', 'resnet101_cbam',
           'resnet152_cbam', 'CBAM_Resnext']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class channel_Weight(nn.Module):
    def __init__(self, inchannel, ratio=16, pool_type=["avg", "max"]):
        super(channel_Weight, self).__init__()
        self.fc = nn.Sequential(Flatten(),
                                nn.Linear(inchannel, inchannel // ratio, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(inchannel // ratio, inchannel, bias=False))
        self.pool = pool_type

    def forward(self, x):
        sum = None
        for i in self.pool:
            if i == "avg":
                avg = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                # C*H*W---->1*H*W
                feature = self.fc(avg)
            elif i == "max":
                max = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                feature = self.fc(max)
            if sum is None:
                sum = feature
            else:
                sum += feature

        weight = F.sigmoid(sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return weight * x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class Spatial_weight(nn.Module):
    def __init__(self):
        super(Spatial_weight, self).__init__()
        self.pool = ChannelPool()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3),
                                  nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True))

    def forward(self, x):
        spatial = self.pool(x)
        weight = self.conv(spatial)
        weight = F.sigmoid(weight)
        return x * weight


class CBAM(nn.Module):
    def __init__(self, inchannel, ratio=16, pool_type=["avg", "max"]):
        super(CBAM, self).__init__()
        self.channnel_Weight = channel_Weight(inchannel, ratio=ratio, pool_type=pool_type)
        self.Spatial_weight = Spatial_weight()

    def forward(self, x):
        x = self.channnel_Weight(x)
        x = self.Spatial_weight(x)
        return x


class block(nn.Module):
    def __init__(self, inchannels, outchannels, stride=1, shortcut=False):
        super(block, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=inchannels, out_channels=outchannels // 2, kernel_size=1, stride=stride),
            nn.BatchNorm2d(outchannels // 2),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=outchannels // 2, out_channels=outchannels // 2, kernel_size=3, padding=1, groups=32,
                      stride=1),
            nn.BatchNorm2d(outchannels // 2),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=outchannels // 2, out_channels=outchannels, kernel_size=1, stride=1),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True)
        )
        self.layer4 = CBAM(outchannels)
        self.shortcut = shortcut
        if shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=inchannels, out_channels=outchannels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(outchannels)
            )

    def forward(self, x):
        if self.shortcut:
            resduial = self.shortcut(x)
        else:
            resduial = x
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x + resduial
        return x


class CBAM_Resnext(nn.Module):
    def __init__(self, layer_number, num_class):
        super(CBAM_Resnext, self).__init__()
        if layer_number == 50:
            self.layer = [3, 4, 6, 3]
        elif layer_number == 101:
            self.layer = [3, 4, 23, 3]
        elif layer_number == 152:
            self.layer = [3, 8, 36, 3]
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        )
        self.layer2 = self.make_layer(64, 256, 1, self.layer[0])
        self.layer3 = self.make_layer(256, 512, 2, self.layer[1])
        self.layer4 = self.make_layer(512, 1024, 2, self.layer[2])
        self.layer5 = self.make_layer(1024, 2048, 2, self.layer[2])
        self.layer6 = nn.AvgPool2d(kernel_size=7, stride=1)
        self.layer7 = nn.Linear(2048, num_class)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def make_layer(self, inchannels, outchannels, stride, num):
        layers = []
        layers.append(block(inchannels, outchannels, stride=stride, shortcut=True))
        for i in range(1, num):
            layers.append(block(outchannels, outchannels, stride=1, shortcut=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = torch.flatten(x, 1)
        print(x.size())
        x = self.layer7(x)
        return F.softmax(x, dim=1)


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def resnet18_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet18'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet34_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet34'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet50_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet50'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet101_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet101'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


def resnet152_cbam(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        pretrained_state_dict = model_zoo.load_url(model_urls['resnet152'])
        now_state_dict = model.state_dict()
        now_state_dict.update(pretrained_state_dict)
        model.load_state_dict(now_state_dict)
    return model


if __name__ == '__main__':
    device = torch.device("cuda:0")
    Xception = CBAM_Resnext(50, 2).to(device)
    # from torchsummary import summary
    # summary(Xception,(3,224,224))
