# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2

from nanotrack.utils.bbox import corner2center, \
        Center, center2corner, Corner


class Augmentation:
    def __init__(self, shift, scale, blur, flip, color):
        self.shift = shift
        self.scale = scale
        self.blur = blur
        self.flip = flip
        self.color = color
        self.rgbVar = np.array(
            [[-0.55919361,  0.98062831, - 0.41940627],
             [1.72091413,  0.19879334, - 1.82968581],
             [4.64467907,  4.73710203, 4.88324118]], dtype=np.float32)#定义bgr三通道的方差

    @staticmethod
    def random():
        return np.random.random() * 2 - 1.0

    def _crop_roi(self, image, bbox, out_sz, padding=(0, 0, 0)):#裁剪感兴趣区域
        bbox = [float(x) for x in bbox]#拿到bbox框xywh
        a = (out_sz-1) / (bbox[2]-bbox[0])#ab为xy方向上的缩放因子
        b = (out_sz-1) / (bbox[3]-bbox[1])
        c = -a * bbox[0]#cd为平移因子
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float64)
        crop = cv2.warpAffine(image, mapping, (out_sz, out_sz),#仿射变换生成裁切图
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=padding)
        return crop

    def _blur_aug(self, image):
        def rand_kernel():
            sizes = np.arange(5, 46, 2)
            size = np.random.choice(sizes)#随机模糊框大小
            kernel = np.zeros((size, size))
            c = int(size/2)#中心点
            wx = np.random.random()#取一个随机值
            kernel[:, c] += 1. / size * wx#水平中心线上有值
            kernel[c, :] += 1. / size * (1-wx)#垂直中心线上有值
            return kernel
        kernel = rand_kernel()#随机一个模糊核尺寸
        image = cv2.filter2D(image, -1, kernel)#自定义卷积核做卷积，中心线模糊，-1代表输入输出尺度相同
        return image

    def _color_aug(self, image):
        offset = np.dot(self.rgbVar, np.random.randn(3, 1))#矩阵乘法计算一个随机偏移量，结果为3*1的矩阵
        offset = offset[::-1]  # bgr 2 rgb
        offset = offset.reshape(3)#转一维数组
        image = image - offset
        return image

    def _gray_aug(self, image):#bgr转灰度，灰度再转bgr
        grayed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(grayed, cv2.COLOR_GRAY2BGR)
        return image

    def _shift_scale_aug(self, image, bbox, crop_bbox, size):#平移/缩放数据增强
        im_h, im_w = image.shape[:2]#原图wh

        # adjust crop bounding box 改图片
        crop_bbox_center = corner2center(crop_bbox)#裁切边界框xyxy转中心点xywh
        if self.scale:
            scale_x = (1.0 + Augmentation.random() * self.scale)#随机xy缩放比例
            scale_y = (1.0 + Augmentation.random() * self.scale)
            h, w = crop_bbox_center.h, crop_bbox_center.w
            scale_x = min(scale_x, float(im_w) / w)
            scale_y = min(scale_y, float(im_h) / h)#不超过图像尺寸和裁切边界的比例，防止超出原图
            crop_bbox_center = Center(crop_bbox_center.x,
                                      crop_bbox_center.y,
                                      crop_bbox_center.w * scale_x,
                                      crop_bbox_center.h * scale_y)#更新缩放后的中心点xywh

        crop_bbox = center2corner(crop_bbox_center)#缩放后的xywh转回xyxy
        if self.shift:
            sx = Augmentation.random() * self.shift
            sy = Augmentation.random() * self.shift#随机平移量

            x1, y1, x2, y2 = crop_bbox

            sx = max(-x1, min(im_w - 1 - x2, sx))
            sy = max(-y1, min(im_h - 1 - y2, sy))#平移后不超过边界

            crop_bbox = Corner(x1 + sx, y1 + sy, x2 + sx, y2 + sy)#更新xyxy

        # adjust target bounding box 改标签
        x1, y1 = crop_bbox.x1, crop_bbox.y1
        bbox = Corner(bbox.x1 - x1, bbox.y1 - y1,
                      bbox.x2 - x1, bbox.y2 - y1)#标签平移量调整

        if self.scale:
            bbox = Corner(bbox.x1 / scale_x, bbox.y1 / scale_y,
                          bbox.x2 / scale_x, bbox.y2 / scale_y)#标签缩放量调整

        image = self._crop_roi(image, crop_bbox, size)#切图输出
        return image, bbox

    def _flip_aug(self, image, bbox):#翻转数据增强
        image = cv2.flip(image, 1)#翻转，1为水平翻转、0为垂直翻转、-1为水平和垂直同时翻转
        width = image.shape[1]#获取当前w
        bbox = Corner(width - 1 - bbox.x2, bbox.y1,
                      width - 1 - bbox.x1, bbox.y2)#水平翻转，x坐标也一并翻转
        return image, bbox

    def __call__(self, image, bbox, size, gray=False):#函数调用
        shape = image.shape
        crop_bbox = center2corner(Center(shape[0]//2, shape[1]//2,
                                         size-1, size-1))#xywh转xyxy
        # gray augmentation灰度增强
        if gray:
            image = self._gray_aug(image)

        # shift scale augmentation 平移缩放增强
        image, bbox = self._shift_scale_aug(image, bbox, crop_bbox, size)

        # color augmentation 颜色增强
        if self.color > np.random.random():
            image = self._color_aug(image)

        # blur augmentation模糊增强
        if self.blur > np.random.random():
            image = self._blur_aug(image)

        # flip augmentation翻转增强
        if self.flip and self.flip > np.random.random():
            image, bbox = self._flip_aug(image, bbox)
        return image, bbox
