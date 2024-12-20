# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn.functional as F


def xcorr_slow(x, kernel):
    """for loop to calculate cross correlation, slow version 慢速计算相关性
    """
    batch = x.size()[0]
    out = []
    for i in range(batch):
        px = x[i]
        pk = kernel[i]
        px = px.view(1, -1, px.size()[1], px.size()[2])
        pk = pk.view(1, -1, pk.size()[1], pk.size()[2])
        po = F.conv2d(px, pk)
        out.append(po)
    out = torch.cat(out, 0)
    return out 

def xcorr_fast(x, kernel):
    """group conv2d to calculate cross correlation, fast version 卷积快速计算相关性
    """
    batch = kernel.size()[0]
    pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
    px = x.view(1, -1, x.size()[2], x.size()[3])
    po = F.conv2d(px, pk, groups=batch)
    po = po.view(batch, -1, po.size()[2], po.size()[3]) 
    return po 

def xcorr_depthwise(x, kernel): 
    """depthwise cross correlation 深度卷积相关性
    """
    batch = kernel.size(0)
    channel = kernel.size(1)  
    x = x.view(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3)) 
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3)) 

    return out

def xcorr_pixelwise(x,kernel): #z=kernel 
    """Pixel-wise correlation (implementation by matrix multiplication)逐像素计算相关性
    The speed is faster because the computation is vectorized"""
    b, c, h, w = x.size() 
    kernel_mat = kernel.view((b, c, -1)).transpose(1, 2)  # (b, hz * wz, c)
    x_mat = x.view((b, c, -1))  # (b, c, hx * wx)
    return torch.matmul(kernel_mat, x_mat).view((b, -1, h, w))  # (b, hz * wz, hx * wx) --> (b, hz * wz, hx, wx)
