# -*- coding: utf-8 -*-
# @Time    : 18-3-27 下午2:26
# @Author  : junhao.li

import torch
import torch.nn as nn
from torch.autograd import Variable

import math

use_cuda = 0
torch.manual_seed(2019)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=padding, dilation=dilation,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
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

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Resnet_GRU_model(nn.Module):
    """
    CNN + RNN as the Encoder, the CNN part is self defined residual net,
    and the RNN part we use two layers GRU, replace average pooling
    """

    def __init__(self, bottleneck, num_class, rnn_hidden_size=200, dropout=0):
        super(Resnet_GRU_model, self).__init__()
        self.rnn_hidden_size = rnn_hidden_size
        self.inplanes = 64
        # Module list
        # [3 * 32 * 280] ==> [64 * 16 * 140]
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=1, stride=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # after layer_1: [128 * 16 * 140]
        self.layer_1 = self._make_layer(bottleneck, 32, blocks=3, stride=1, dilation=1)
        # after layer_2: [256 * 8 * 70]
        self.layer_2 = self._make_layer(bottleneck, 64, blocks=4, stride=2, dilation=2)
        # after layer_3: [256 * 8 * 70]
        self.layer_3 = self._make_layer(bottleneck, 64, blocks=5, stride=1, dilation=2)
        # after layer_4: [256 * 1 * 70]
        # self.layer_4 = nn.AvgPool2d(kernel_size=(8, 1), padding=(0, 0), stride=1)
        self.layer_4 = nn.MaxPool2d(kernel_size=(64, 1), padding=(0, 0), stride=1)
        # RNN
        self.gru_1 = nn.GRU(input_size=256,
                            hidden_size=self.rnn_hidden_size,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=True)

        self.gru_2 = nn.GRU(input_size=self.rnn_hidden_size * 2,
                            hidden_size=self.rnn_hidden_size,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=True)

        # fully connected layers
        self.fc1 = nn.Linear(1856, num_class)

        # weight initiation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                *[nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                  nn.BatchNorm2d(planes * block.expansion)])

        layers = list()
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1, dilation=dilation, downsample=None))

        return nn.Sequential(*layers)

    def init_rnn_weight(self, batch_size, rnn_hidden_size):
        h = Variable(torch.rand(2, batch_size, rnn_hidden_size), requires_grad=True)
        h = h.cuda() if use_cuda else h
        return h

    def forward(self, x, h1=None, h2=None):

        batch_size = x.size()[0]
        
        h1 = self.init_rnn_weight(batch_size, self.rnn_hidden_size) if h1 is None else h1
        rnn_x = x.squeeze(1)
        print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)        
        print(x.shape)

        x = self.relu(x)
        
        # added
        # layer_1 ~ layer_4: 
        x = self.layer_1(x)
        print(x.shape)
        x = self.layer_2(x)
        print(x.shape)
        x = self.layer_3(x)
        print(x.shape)
        x = self.layer_4(x)
        print("layer_4",x.shape)
        
        # after average pooling the feature map: [batch_size, 256, 1, 70]
        # remove dim = 1，as RNN model has this limit, only support three dim
        # [batch_size, num_feature_map, 1, w]  ==> [batch_size, num_feature_map, w]
        x = x.squeeze(2)
        print("squeeze",x.shape)
        x = x.transpose(1, 2)  # [batch_size, w, num_feature_map]
        cnn_feature = x
        print("transpose",x.shape)
        # after the first bidirectional GRU module, [batch_size, 70, 256]  ==> [batch_size, 70, 256 * 2]
        # after the second  bidirectional GRU module, [batch_size, 70, 256 * 2]  ==> [batch_size, 70, 256 * 2]
        x, rnn_h1 = self.gru_1(x, h1)
        # residual = x
        print(x.shape)
        x, rnn_h2 = self.gru_2(x, rnn_h1)
        print(x.shape)
        # x = x + residual

        rnn_feature = x.transpose(1, 2)

        x = torch.cat((cnn_feature, rnn_feature), dim=2)
        # fully connected layers

        x = self.fc1(x)
        return x, rnn_h1, rnn_h2


