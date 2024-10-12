# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from nanotrack.core.config import cfg
from nanotrack.models.iou_loss import linear_iou


def get_cls_loss(pred, label, select):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select) #指定维度，提取预测和标签中对应的值
    label = torch.index_select(label, 0, select)
    return F.nll_loss(pred, label)#计算负对数似然损失

def select_cross_entropy_loss(pred, label): #预测值和标签计算交叉熵
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero().squeeze().cuda() #找标签中为1的正样本
    neg = label.data.eq(0).nonzero().squeeze().cuda() #找标签中为0的负样本
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5 #各取0.5加权正负样本

def weight_l1_loss(pred_loc, label_loc, loss_weight):
    if cfg.BAN.BAN: 
        diff = (pred_loc - label_loc).abs()
        diff = diff.sum(dim=1)
    else:
        diff = None
    loss = diff * loss_weight
    return loss.sum().div(pred_loc.size()[0])


def select_iou_loss(pred_loc, label_loc, label_cls):
    label_cls = label_cls.reshape(-1)
    pos = label_cls.data.eq(1).nonzero().squeeze().cuda() #找正标签的索引

    pred_loc = pred_loc.permute(0, 2, 3, 1).reshape(-1, 4)
    pred_loc = torch.index_select(pred_loc, 0, pos) #找正标签索引下的预测值

    label_loc = label_loc.permute(0, 2, 3, 1).reshape(-1, 4)
    label_loc = torch.index_select(label_loc, 0, pos) #找正标签索引下的标签值

    return linear_iou(pred_loc, label_loc)#自定义了iou损失的计算
