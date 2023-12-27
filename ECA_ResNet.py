#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

"""
@File    : ECA_ResNet.py
@Author  : cao xu
@Time    : 2023/12/27 17:09
"""
import torch.nn as nn
import math
import torch.nn.functional as F


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ECABasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        super(ECABasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.eca = eca_layer(planes, k_size)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ECABottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        super(ECABottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.eca = eca_layer(planes * 4, k_size)
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
        out = self.eca(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, k_size=[3, 3, 3, 3]):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], int(k_size[0]))
        self.layer2 = self._make_layer(block, 128, layers[1], int(k_size[1]), stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], int(k_size[2]), stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], int(k_size[3]), stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, k_size, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, k_size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, k_size=k_size))

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


def eca_resnet18(k_size=[3, 3, 3, 3], num_classes=1_000, pretrained=False):
    """Constructs a ResNet-18 model.
    Args:
        k_size: Adaptive selection of kernel size
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes:The classes of classification
    """
    model = ResNet(ECABasicBlock, [2, 2, 2, 2], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def eca_resnet34(k_size=[3, 3, 3, 3], num_classes=1_000, pretrained=False):
    """Constructs a ResNet-34 model.
    Args:
        k_size: Adaptive selection of kernel size
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes:The classes of classification
    """
    model = ResNet(ECABasicBlock, [3, 4, 6, 3], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def eca_resnet50(k_size=[3, 3, 3, 3], num_classes=1000, pretrained=False):
    """Constructs a ResNet-50 model.
    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    print("Constructing eca_resnet50......")
    model = ResNet(ECABottleneck, [3, 4, 6, 3], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def eca_resnet101(k_size=[3, 3, 3, 3], num_classes=1_000, pretrained=False):
    """Constructs a ResNet-101 model.
    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(ECABottleneck, [3, 4, 23, 3], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def eca_resnet152(k_size=[3, 3, 3, 3], num_classes=1_000, pretrained=False):
    """Constructs a ResNet-152 model.
    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(ECABottleneck, [3, 8, 36, 3], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


class BN_Conv2d(nn.Module):
    """
    BN_CONV_RELU
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=False):
        super(BN_Conv2d, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return F.relu(self.seq(x))


class ResNeXt(nn.Module):
    """
    ResNeXt builder
    """

    def __init__(self, layers: object, cardinality, group_depth, num_classes) -> object:
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.channels = 64
        self.conv1 = BN_Conv2d(3, self.channels, 7, stride=2, padding=3)
        d1 = group_depth
        self.conv2 = self.___make_layers(d1, layers[0], stride=1)
        d2 = d1 * 2
        self.conv3 = self.___make_layers(d2, layers[1], stride=2)
        d3 = d2 * 2
        self.conv4 = self.___make_layers(d3, layers[2], stride=2)
        d4 = d3 * 2
        self.conv5 = self.___make_layers(d4, layers[3], stride=2)
        self.fc = nn.Linear(self.channels, num_classes)   # 224x224 input size

    def ___make_layers(self, d, blocks, stride):
        strides = [stride] + [1] * (blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResNeXt_Block(self.channels, self.cardinality, d, stride))
            self.channels = self.cardinality*d*2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = F.max_pool2d(out, 3, 2, 1)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = F.softmax(self.fc(out))
        return out


class ResNeXt_Block(nn.Module):
    """
    ResNeXt block with group convolutions
    """

    def __init__(self, in_chnls, cardinality, group_depth, stride):
        super(ResNeXt_Block, self).__init__()
        self.group_chnls = cardinality * group_depth
        self.conv1 = BN_Conv2d(in_chnls, self.group_chnls, 1, stride=1, padding=0)
        self.conv2 = BN_Conv2d(self.group_chnls, self.group_chnls, 3, stride=stride, padding=1, groups=cardinality)
        self.conv3 = nn.Conv2d(self.group_chnls, self.group_chnls*2, 1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(self.group_chnls*2)
        self.short_cut = nn.Sequential(
            nn.Conv2d(in_chnls, self.group_chnls*2, 1, stride, 0, bias=False),
            nn.BatchNorm2d(self.group_chnls*2)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn(self.conv3(out))
        out += self.short_cut(x)
        return F.relu(out)


def eca_resNeXt50_32x4d(num_classes=1000):
    return ResNeXt([3, 4, 6, 3], 32, 4, num_classes)


def eca_resNeXt101_32x4d(num_classes=1000):
    return ResNeXt([3, 4, 23, 3], 32, 4, num_classes)


def eca_resNeXt101_64x4d(num_classes=1000):
    return ResNeXt([3, 4, 23, 3], 64, 4, num_classes)