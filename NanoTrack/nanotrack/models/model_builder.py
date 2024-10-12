# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch 
import torch.nn as nn
import torch.nn.functional as F

from nanotrack.core.config import cfg
from nanotrack.models.loss import select_cross_entropy_loss, select_iou_loss
from nanotrack.models.backbone import get_backbone
from nanotrack.models.head import get_ban_head
from nanotrack.models.neck import get_neck 

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone 初始化骨干网络，默认是mobilenetv3_small_v3,作者自己改的，加了一层输出
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer 初始化颈部，默认是AdjustLayer，貌似没用上
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build ban head 初始化头部，默认DepthwiseBAN
        if cfg.BAN.BAN:
            self.ban_head = get_ban_head(cfg.BAN.TYPE,
                                     **cfg.BAN.KWARGS)

    def template(self, z): #用于单次获取模板特征图
        zf = self.backbone(z)
        self.zf = zf #保存下来就不改了
 
    def track(self, x): 

        xf = self.backbone(x)  #搜索图特征输出

        cls, loc = self.ban_head(self.zf, xf) #头部输出，没有颈部？
        # print(cls)
        # print('---------')
        # print(loc)


        return {
                'cls': cls,
                'loc': loc,
               } 

    def log_softmax(self, cls):

        if cfg.BAN.BAN:
            cls = cls.permute(0, 2, 3, 1).contiguous()

            cls = F.log_softmax(cls, dim=3)

        return cls #没有头部就直接softmax输出，有头部就直接输出类别置信度

    #  forward
    def forward(self, data):
        """ only used in training 仅训练使用
        """
        # train mode
        if len(data)>=4: #多输入时判断为训练
            template = data['template'].cuda()
            search = data['search'].cuda()
            label_cls = data['label_cls'].cuda()
            label_loc = data['label_loc'].cuda() #拿到模板和搜索框图和标签，转gpu

            # get feature 分别过骨干网络
            zf = self.backbone(template)
            xf = self.backbone(search)    

            cls, loc = self.ban_head(zf, xf) #头部网络输出

            # cls loss with cross entropy loss 没有头部就直接softmax输出，有头部就直接输出类别置信度
            cls = self.log_softmax(cls) 

            cls_loss = select_cross_entropy_loss(cls, label_cls) #计算类别损失值

            # loc loss with iou loss
            loc_loss = select_iou_loss(loc, label_loc, label_cls) #计算iou损失，默认是1-iou损失
            outputs = {} 

            outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + cfg.TRAIN.LOC_WEIGHT * loc_loss #总损失进行了加权
            outputs['cls_loss'] = cls_loss
            outputs['loc_loss'] = loc_loss

            return outputs  
        
        # test speed 
        else:  #单一输入时判断为测速
        
            xf = self.backbone(data)  #仅搜索框过骨干
            cls, loc = self.ban_head(self.zf, xf) #模板框仅用第一帧计算

            return {
                    'cls': cls,
                    'loc': loc,
                }

