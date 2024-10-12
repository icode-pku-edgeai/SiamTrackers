from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


class Point:
    """
    This class generate points.生成均匀分布的点坐标
    """
    def __init__(self, stride, size, image_center):
        self.stride = stride#步长16
        self.size = size#区域大小15
        self.image_center = image_center#中心坐标127

        self.points = self.generate_points(self.stride, self.size, self.image_center)

    def generate_points(self, stride, size, im_c):
        ori = im_c - size // 2 * stride#初始值 15
        x, y = np.meshgrid([ori + stride * dx for dx in np.arange(0, size)],
                           [ori + stride * dy for dy in np.arange(0, size)])#基于初值 15和stride 16生成网格
        points = np.zeros((2, size, size), dtype=np.float32)#初始化一个np数组
        points[0, :, :], points[1, :, :] = x.astype(np.float32), y.astype(np.float32)#第一个维度区分xy，后面两个维度存储网络坐标值

        return points
