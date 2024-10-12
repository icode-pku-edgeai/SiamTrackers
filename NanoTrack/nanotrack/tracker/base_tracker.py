# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import torch

from nanotrack.core.config import cfg

class BaseTracker(object):#还没实现
    """ Base tracker of single objec tracking
    """
    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox(list): [x, y, width, height]
                        x, y need to be 0-based
        """
        raise NotImplementedError

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        raise NotImplementedError


class SiameseTracker(BaseTracker):
    def get_subwindow(self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: bgr based image 原图
            pos: center position 目标框中心点位置
            model_sz: exemplar size 模板尺寸127*127
            s_z: original size 目标框转正方形的尺寸,用于缩放
            avg_chans: channel average 原图hw两个通道上的
        """
        if isinstance(pos, float):
            pos = [pos, pos] #原图中心点xy
        sz = original_sz #目标框转正方形的尺寸
        im_sz = im.shape #原始图像的尺寸,例如720*1280
        c = (original_sz + 1) / 2 #正方形的半边长
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)#中心点不变，中心点+正方形半边长求出左侧点
        context_xmax = context_xmin + sz - 1#左边点+正方形边长求右侧点
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1 #计算xyxy
        left_pad = int(max(0., -context_xmin))#计算上下左右是否超边界
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad #只用了左边和上边进行填充，用于裁出来不够，padding补齐
        
        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]): #非全零,就要做填充，用rgb三通道的均值填充
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad: 
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad: 
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                             int(context_xmin):int(context_xmax + 1), :]
        else: #全零说明不用填充
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                          int(context_xmin):int(context_xmax + 1), :]#用调整后的xyxy将图切成正方形

        if not np.array_equal(model_sz, original_sz): #resize到127*127
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1) #转RGB
        im_patch = im_patch[np.newaxis, :, :, :] #加根batch轴
        im_patch = im_patch.astype(np.float32) #转fp32
        im_patch = torch.from_numpy(im_patch)#转numpy
        if cfg.CUDA:
            im_patch = im_patch.cuda()#转cuda
        return im_patch
