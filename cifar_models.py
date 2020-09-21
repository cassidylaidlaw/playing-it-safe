'''
TAKEN FROM https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py

Properly implemented ResNet-s for CIFAR10 as described in paper [1].
The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.
Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m
which this implementation indeed has.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math

import utils

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56',
           'resnet110', 'resnet1202', 'resnet20w', 'resnet20drop',
           'WideResNet28x']


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        try:
            init.kaiming_normal_(m.weight)
        except AttributeError:
            init.kaiming_normal(m.weight)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=math.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    shortcut: nn.Module

    def __init__(self, in_planes, planes, stride=1, option='A',
                 bias=False, dropout=0):
        super(BasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=bias)

        self.dropout = dropout

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                # For CIFAR10 ResNet paper uses option A.
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], [0, 0, 0, 0, planes//4, planes//4], "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )
            elif option == 'wide':
                self.shortcut = \
                    nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True)

    def forward(self, x):
        out = utils.activation(self.bn1(self.conv1(x)))
        if self.dropout > 0:
            out = F.dropout2d(out, self.dropout)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = utils.activation(out)
        return out


class WideBasicBlock(BasicBlock):
    def __init__(self, *args, **kwargs):
        kwargs['option'] = 'wide'
        super().__init__(*args, **kwargs)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, width=16, bias=False,
                 dropout=0):
        super(ResNet, self).__init__()
        self.in_planes = width
        self.bias = bias
        self.dropout = dropout
        self.num_layers = sum(num_blocks)

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, width * 2, num_blocks[1],
                                       stride=2)
        self.layer3 = self._make_layer(block, width * 4, num_blocks[2],
                                       stride=2)
        self.linear = nn.Linear(width * 4, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, bias=self.bias,
                                dropout=self.dropout))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = utils.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(**kwargs):
    return ResNet(BasicBlock, [3, 3, 3], **kwargs)


def resnet32(**kwargs):
    return ResNet(BasicBlock, [5, 5, 5], **kwargs)


def resnet44(**kwargs):
    return ResNet(BasicBlock, [7, 7, 7], **kwargs)


def resnet56(**kwargs):
    return ResNet(BasicBlock, [9, 9, 9], **kwargs)


def resnet110(**kwargs):
    return ResNet(BasicBlock, [18, 18, 18], **kwargs)


def resnet152(**kwargs):
    return ResNet(BasicBlock, [25, 25, 25], **kwargs)


def resnet200(**kwargs):
    return ResNet(BasicBlock, [33, 33, 33], **kwargs)


def resnet1202(**kwargs):
    return ResNet(BasicBlock, [200, 200, 200], **kwargs)


def resnet20drop(dropout):
    def resnet20(**kwargs):
        return ResNet(BasicBlock, [3, 3, 3], dropout=dropout,
                      **kwargs)
    return resnet20


def resnet20w(width):
    def resnet20(**kwargs):
        return ResNet(BasicBlock, [3, 3, 3], width=width, **kwargs)
    return resnet20


def WideResNet28x(width, dropout=0):
    def wide_resnet(**kwargs):
        model = ResNet(WideBasicBlock, [4, 4, 4], width=width*16, bias=True,
                       dropout=dropout, **kwargs)
        return model
    return wide_resnet
