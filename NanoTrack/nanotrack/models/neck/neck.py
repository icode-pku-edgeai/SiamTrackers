# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn

class AdjustLayer(nn.Module):#单级颈部
    def __init__(self, in_channels, out_channels):
        super(AdjustLayer, self).__init__()

        self.in_channels=in_channels

        self.out_channels=out_channels 

        self.downsample = nn.Sequential(#降采样，用1*1的卷积把输入输出通道输沟通起来
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            ) 

    def forward(self, x):

        if self.in_channels != self.out_channels:
            x = self.downsample(x) 

        if x.size(3) < 16: 
            l = 2
            r = l + 4 
            x = x[:, :, l:r, l:r]
        return x 

class AdjustAllLayer(nn.Module):#多级颈部
    def __init__(self, in_channels, out_channels):
        super(AdjustAllLayer, self).__init__()
        self.num = len(out_channels)
        if self.num == 1:
            self.downsample = AdjustLayer(in_channels[0], out_channels[0])
        else:
            for i in range(self.num):
                self.add_module('downsample'+str(i+2),
                                AdjustLayer(in_channels[i], out_channels[i]))

    def forward(self, features):
        if self.num == 1:
            return self.downsample(features)
        else:
            out = []
            for i in range(self.num):
                adj_layer = getattr(self, 'downsample'+str(i+2))
                out.append(adj_layer(features[i]))
            return out 
