from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from nanotrack.core.config import cfg
from nanotrack.utils.bbox import corner2center
from nanotrack.utils.point import Point


class PointTarget:
    def __init__(self,):
        self.points = Point(cfg.POINT.STRIDE, cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.SEARCH_SIZE//2)
        #基于定义的输出尺寸、步长、搜索框大小定义一个网格，三个参数分别是16、15、127

    def __call__(self, target, size, neg=False):#作为函数，使用()调用时使用

        # -1 ignore 初始化、0 negative 异类、1 positive同类
        cls = -1 * np.ones((size, size), dtype=np.int64)#创建一个int64的全是-1的二维数组，分类
        delta = np.zeros((4, size, size), dtype=np.float32)#fp32的4位输出的三维输出，回归

        def select(position, keep_num=16):#
            num = position[0].shape[0]#统计数量
            if num <= keep_num:#不足16个就全部返回
                return position, num
            slt = np.arange(num)
            np.random.shuffle(slt)
            slt = slt[:keep_num]#超过16个就洗牌后随机裁16个返回
            return tuple(p[slt] for p in position), keep_num

        tcx, tcy, tw, th = corner2center(target)#目标框bbox的xyxy转中心点xywh
        points = self.points.points#拿到初始化的网格点

        if neg:#默认是关闭的，打开时为全负样本的状态
            neg = np.where(np.square(tcx - points[0]) / np.square(tw / 4) +
                           np.square(tcy - points[1]) / np.square(th / 4) < 1)#返回所有不是同类的索引
            neg, neg_num = select(neg, cfg.TRAIN.NEG_NUM)#随机抽16个
            cls[neg] = 0#全部定义为异类

            return cls, delta#结束异类的点阵

        delta[0] = points[0] - target[0]
        delta[1] = points[1] - target[1]
        delta[2] = target[2] - points[0]
        delta[3] = target[3] - points[1]#定义点阵到目标框bbox的距离

        # ellipse label
        pos = np.where(np.square(tcx - points[0]) / np.square(tw / 4) +
                       np.square(tcy - points[1]) / np.square(th / 4) < 1)#判断点阵与目标框重合的索引为正
        neg = np.where(np.square(tcx - points[0]) / np.square(tw / 2) +
                       np.square(tcy - points[1]) / np.square(th / 2) > 1)#判断点阵与目标框不重合的索引为负
        
        # sampling
        pos, pos_num = select(pos, cfg.TRAIN.POS_NUM)#准备正样本，16个一组
        neg, neg_num = select(neg, cfg.TRAIN.TOTAL_NUM - cfg.TRAIN.POS_NUM)#准备负样本，64-16的负样本量

        cls[pos] = 1#根据索引定义好正负样本
        cls[neg] = 0

        return cls, delta
