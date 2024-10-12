
# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os
import time
import math
import json
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader 

from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.distributed import DistributedSampler

import sys 
sys.path.append(os.getcwd()) 

from nanotrack.utils.lr_scheduler import build_lr_scheduler
from nanotrack.utils.log_helper import init_log, print_speed, add_file_handler
from nanotrack.utils.distributed import  new_dist_init, DistModule, reduce_gradients,average_reduce, get_rank, get_world_size
from nanotrack.utils.model_load import load_pretrain, restore_from
from nanotrack.utils.average_meter import AverageMeter
from nanotrack.utils.misc import describe, commit
from nanotrack.models.model_builder import ModelBuilder
from nanotrack.datasets.dataset import BANDataset
from nanotrack.core.config import cfg

logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='nanotrack') 
parser.add_argument('--cfg', type=str, default='./models/config/configv3.yaml',help='configuration of tracking')
parser.add_argument('--seed', type=int, default=123456, help='random seed')
parser.add_argument('--local_rank', type=int, default=0, help='compulsory for pytorch launcer')#PyTorch启动器的必需条件或要素
args = parser.parse_args()


def seed_torch(seed=0):
    random.seed(seed)# 设置 Python 标准库中的随机数生成器的种子
    os.environ['PYTHONHASHSEED'] = str(seed)#设置 Python 的哈希随机化种子
    np.random.seed(seed)#设置 NumPy 的随机数生成器的种子
    torch.manual_seed(seed) #设置 PyTorch CPU 端的随机数生成器的种子
    torch.cuda.manual_seed(seed)#设置 GPU 端的随机数生成器的种子
    torch.backends.cudnn.benchmark = False#禁用 cuDNN 的自动调优功能，放弃cudnn自动选择最优算法加速计算，以便得到重复性结果   
    torch.backends.cudnn.deterministic = True#启用 cuDNN 的确定性模式

def build_data_loader():
    logger.info("build train dataset")
    # train_dataset
    if cfg.BAN.BAN:#头部网络
        train_dataset = BANDataset()#根据头部网络准备数据集
        logger.info("build BANDataset done")
        # print(train_dataset[0]['template'].shape)
        # print(train_dataset[0]['search'].shape)
    logger.info("build dataset done")

    train_sampler = None
    if get_world_size() > 1:#初始设置的是1，应该是gpu数量
        train_sampler = DistributedSampler(train_dataset)#数据集子集采样器
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                            #   num_workers=cfg.TRAIN.NUM_WORKERS,
                              num_workers=2,
                              pin_memory=True,
                              sampler=train_sampler)
    return train_loader

def build_opt_lr(model, current_epoch=0):#配置优化器和学习率
    for param in model.backbone.parameters():
        param.requires_grad = False#梯度首先全关
    for m in model.backbone.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()#bn转验证模式
    if current_epoch >= cfg.BACKBONE.TRAIN_EPOCH:#超过10后，才优化骨干
        for layer in cfg.BACKBONE.TRAIN_LAYERS:#'features'
            for param in getattr(model.backbone, layer).parameters():#特征有关的backbone梯度全部打开
                param.requires_grad = True
            for m in getattr(model.backbone, layer).modules():#bn转训练模式
                if isinstance(m, nn.BatchNorm2d):
                    m.train()

    trainable_params = []
    trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                           model.backbone.parameters()), #需要梯度的参数统计
                          'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}] #0.1*0.005

    if cfg.ADJUST.ADJUST:#颈部参数添加
        trainable_params += [{'params': model.neck.parameters(),
                              'lr': cfg.TRAIN.BASE_LR}]

    trainable_params += [{'params': model.ban_head.parameters(),#头部参数添加
                          'lr': cfg.TRAIN.BASE_LR}]
    #设置SGD优化器
    optimizer = torch.optim.SGD(trainable_params,
                                momentum=cfg.TRAIN.MOMENTUM,#0.9
                                weight_decay=cfg.TRAIN.WEIGHT_DECAY)#0.0001

    lr_scheduler = build_lr_scheduler(optimizer, epochs=cfg.TRAIN.EPOCH)#建立学习率表，#10
    lr_scheduler.step(cfg.TRAIN.START_EPOCH)#从0开始
    return optimizer, lr_scheduler


def log_grads(model, tb_writer, tb_index):#模型、tensorboard写入器和索引
    def weights_grads(model):
        grad = {}
        weights = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad[name] = param.grad
                weights[name] = param.data 
        return grad, weights#收集梯度和权重

    grad, weights = weights_grads(model)
    feature_norm, head_norm = 0, 0 #记录骨干和头部的l2范数
    for k, g in grad.items():
        _norm = g.data.norm(2)
        weight = weights[k]
        w_norm = weight.norm(2)
        if 'feature' in k:
            feature_norm += _norm ** 2#计算l2范数
        else:
            head_norm += _norm ** 2

        tb_writer.add_scalar('grad_all/'+k.replace('.', '/'),#写入tensorboard
                             _norm, tb_index)
        tb_writer.add_scalar('weight_all/'+k.replace('.', '/'),
                             w_norm, tb_index)
        tb_writer.add_scalar('w-g/'+k.replace('.', '/'),
                             w_norm/(1e-20 + _norm), tb_index)
    tot_norm = feature_norm + head_norm
    tot_norm = tot_norm ** 0.5
    feature_norm = feature_norm ** 0.5
    head_norm = head_norm ** 0.5

    tb_writer.add_scalar('grad/tot', tot_norm, tb_index)#所有梯度
    tb_writer.add_scalar('grad/feature', feature_norm, tb_index)#特征梯度
    tb_writer.add_scalar('grad/head', head_norm, tb_index)#头部梯度


def train(train_loader, model, optimizer, lr_scheduler, tb_writer):
    cur_lr = lr_scheduler.get_cur_lr()#获取当前学习率
    rank = get_rank()#线程相关，0

    average_meter = AverageMeter()#计算和存储平均值和当前值

    def is_valid_number(x):#判断是否有效：nan/inf/大于10000
        return not(math.isnan(x) or math.isinf(x) or x > 1e4)

    world_size = get_world_size()#1，分布式训练
    num_per_epoch = len(train_loader.dataset) // \
        cfg.TRAIN.EPOCH // (cfg.TRAIN.BATCH_SIZE * world_size)#每代数据量=总数据量20000000/训练代数50/batch大小64
    start_epoch = cfg.TRAIN.START_EPOCH
    epoch = start_epoch 

    if not os.path.exists(cfg.TRAIN.SNAPSHOT_DIR) and \
            get_rank() == 0:
        os.makedirs(cfg.TRAIN.SNAPSHOT_DIR)#创建存储的文件夹

    logger.info("model\n{}".format(describe(model.module)))
    end = time.time()#记录时间
    for idx, data in enumerate(train_loader):#单个遍历数据集
        # print(data['template'].shape)
        # print(data['search'].shape)
        if epoch != idx // num_per_epoch + start_epoch:#一个epoch做一次保存
            epoch = idx // num_per_epoch + start_epoch#定义当前epoch

            if get_rank() == 0:#初代存储
                torch.save(
                        {'epoch': epoch,
                         'state_dict': model.module.state_dict(),
                         'optimizer': optimizer.state_dict()},
                        cfg.TRAIN.SNAPSHOT_DIR+'/checkpoint_e%d.pth' % (epoch))

            if epoch == cfg.TRAIN.EPOCH:#完成即结束
                return

            if cfg.BACKBONE.TRAIN_EPOCH == epoch:#10代后开始训练骨干
                logger.info('start training backbone.')
                optimizer, lr_scheduler = build_opt_lr(model.module, epoch)
                logger.info("model\n{}".format(describe(model.module)))

            lr_scheduler.step(epoch)#学习率递进 
            cur_lr = lr_scheduler.get_cur_lr()#获取当前学习率
            logger.info('epoch: {}'.format(epoch+1))
        tb_idx = idx + start_epoch * num_per_epoch#tensorboard写入内容
        if idx % num_per_epoch == 0 and idx != 0:
            for idx, pg in enumerate(optimizer.param_groups):
                logger.info('epoch {} lr {}'.format(epoch+1, pg['lr']))
                if rank == 0:
                    tb_writer.add_scalar('lr/group{}'.format(idx+1),
                                         pg['lr'], tb_idx)

        data_time = average_reduce(time.time() - end)#统计时间
        if rank == 0:
            tb_writer.add_scalar('time/data', data_time, tb_idx)#tensorboard写入内容

        outputs = model(data)#数据过模型
        loss = outputs['total_loss']#拿到损失值

        if is_valid_number(loss.data.item()):#损失数据有效的情况下
            optimizer.zero_grad()#优化器梯度清零
            loss.backward()#损失反向传播
            reduce_gradients(model) #多gpu同步梯度下降用

            if rank == 0 and cfg.TRAIN.LOG_GRADS:#默认false，tensorboard写入的操作
                log_grads(model.module, tb_writer, tb_idx)

            # clip gradient
            clip_grad_norm_(model.parameters(), cfg.TRAIN.GRAD_CLIP)#默认值为10，对一组参数的梯度进行范数裁剪，防止梯度爆炸，稳定训练过程
            optimizer.step()#优化器迭代

        batch_time = time.time() - end
        batch_info = {}
        batch_info['batch_time'] = average_reduce(batch_time)
        batch_info['data_time'] = average_reduce(data_time)#记录时间
        for k, v in sorted(outputs.items()):
            batch_info[k] = average_reduce(v.data.item())#记录结果的键值对

        average_meter.update(**batch_info)#更新平均尺度

        if rank == 0:
            for k, v in batch_info.items():
                tb_writer.add_scalar(k, v, tb_idx)#添加tensorboard

            if (idx+1) % cfg.TRAIN.PRINT_FREQ == 0:#默认20代打印一次训练信息
                info = "Epoch: [{}][{}/{}] lr: {:.6f}\n".format(#epoch、当前数据批次、总批次、当前学习率
                            epoch+1, (idx+1) % num_per_epoch,
                            num_per_epoch, cur_lr)
                for cc, (k, v) in enumerate(batch_info.items()):#偶数就水平指表、奇数就换行
                    if cc % 2 == 0:
                        info += ("\t{:s}\t").format(
                                getattr(average_meter, k))
                    else:
                        info += ("{:s}\n").format(
                                getattr(average_meter, k))
                logger.info(info)
                print_speed(idx+1+start_epoch*num_per_epoch, 
                            average_meter.batch_time.avg,
                            cfg.TRAIN.EPOCH * num_per_epoch) #打印训练过程、速度、剩余时间
        end = time.time() 


def main():
    
    rank, world_size = new_dist_init()#初始全局变量，分别为0,1
    
    logger.info("init done")

    # load cfg
    cfg.merge_from_file(args.cfg)
    if rank == 0:
        if not os.path.exists(cfg.TRAIN.LOG_DIR):
            os.makedirs(cfg.TRAIN.LOG_DIR)#日志目录
        init_log('global', logging.INFO)
        if cfg.TRAIN.LOG_DIR:
            add_file_handler('global',
                             os.path.join(cfg.TRAIN.LOG_DIR, 'logs.txt'),
                             logging.INFO)

        logger.info("Version Information: \n{}\n".format(commit()))
        logger.info("config \n{}".format(json.dumps(cfg, indent=4)))

    model = ModelBuilder().cuda().train()#初始化模型

    if cfg.BACKBONE.PRETRAINED:
        cur_path = os.path.dirname(os.path.realpath(__file__))
        backbone_path = os.path.join(cur_path, '../', cfg.BACKBONE.PRETRAINED)
        load_pretrain(model.backbone, backbone_path)#加载骨干预训练权重mobilenet

    if rank == 0 and cfg.TRAIN.LOG_DIR:#tensorboard
        tb_writer = SummaryWriter(cfg.TRAIN.LOG_DIR)
    else:
        tb_writer = None

    train_loader = build_data_loader()#准备数据集

    optimizer, lr_scheduler = build_opt_lr(model,cfg.TRAIN.START_EPOCH)#准备优化器和学习率表

    if cfg.TRAIN.RESUME:#断点存续
        logger.info("resume from {}".format(cfg.TRAIN.RESUME))
        assert os.path.isfile(cfg.TRAIN.RESUME), \
            '{} is not a valid file.'.format(cfg.TRAIN.RESUME)
        model, optimizer, cfg.TRAIN.START_EPOCH = \
            restore_from(model, optimizer, cfg.TRAIN.RESUME)

    # load pretrain
    elif cfg.TRAIN.PRETRAINED:
        load_pretrain(model, cfg.TRAIN.PRETRAINED)#加载预训练权重nanotrack
    dist_model = DistModule(model)#分布式模型

    logger.info(lr_scheduler)#前5个epoch从0.001-0.005线性增加，后面45个epoch逐渐衰减
    logger.info("model prepare done")

    # start training
    train(train_loader, dist_model, optimizer, lr_scheduler, tb_writer)

if __name__ == '__main__':
    seed_torch(args.seed)
    main()
