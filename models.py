import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import utils

from cifar_models import *  # noqa: F401, F403


class DifferentiableNormalizer(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        c = x.shape[1]
        mean_var = Variable(x.data.new(self.mean).view(1, c, 1, 1))
        std_var = Variable(x.data.new(self.std).view(1, c, 1, 1))
        return (x - mean_var) / std_var

    def inverse(self, x):
        c = x.shape[1]
        mean_var = Variable(x.data.new(self.mean).view(1, c, 1, 1))
        std_var = Variable(x.data.new(self.std).view(1, c, 1, 1))
        return (x * std_var) + mean_var


class MnistNet(nn.Module):
    def __init__(self, num_classes=10):
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 9, 3)
        self.conv2 = nn.Conv2d(9, 15, 3)
        self.lin1 = nn.Linear(375, 20)
        self.lin2 = nn.Linear(20, num_classes)

    def forward(self, x):
        out = F.max_pool2d(utils.activation(self.conv1(x)), kernel_size=2)
        out = F.max_pool2d(utils.activation(self.conv2(out)), kernel_size=2)
        out = self.lin2(utils.activation(self.lin1(out.view(out.size(0), -1))))
        return out


class BigMnistNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 32, 5)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.lin1 = nn.Linear(576, 256)
        self.lin2 = nn.Linear(256, num_classes)

    def forward(self, x):
        out = utils.activation(self.conv1(x))
        out = utils.activation(self.conv2(out))
        out = F.max_pool2d(out, kernel_size=2)

        out = utils.activation(self.conv3(out))
        out = utils.activation(self.conv4(out))
        out = F.max_pool2d(out, kernel_size=2)

        out = utils.activation(self.lin1(out.view(out.size(0), -1)))
        out = self.lin2(out)
        return out


def convert_relu_to_softplus(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, nn.Softplus())
        else:
            convert_relu_to_softplus(child)
