# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = ""

__C.CUDA = True 

# ------------------------------------------------------------------------ #
# Training options 
# ------------------------------------------------------------------------ #
__C.TRAIN = CN()

# Number of negative 反例个数
__C.TRAIN.NEG_NUM = 16

# Number of positive 正例个数
__C.TRAIN.POS_NUM = 16

# Number of anchors per image 采样总点数
__C.TRAIN.TOTAL_NUM = 64 


__C.TRAIN.EXEMPLAR_SIZE = 127 #模板框尺寸

__C.TRAIN.SEARCH_SIZE = 255 #搜索框尺寸

__C.TRAIN.BASE_SIZE = 8 #

__C.TRAIN.OUTPUT_SIZE = 25

__C.TRAIN.RESUME = ''

__C.TRAIN.PRETRAINED = ''

__C.TRAIN.LOG_DIR = './logs'

__C.TRAIN.SNAPSHOT_DIR = './snapshot'

__C.TRAIN.EPOCH = 20 #训练代数

__C.TRAIN.START_EPOCH = 0 #起始代数
__C.TRAIN.NUM_CONVS =4 #卷积个数

__C.TRAIN.BATCH_SIZE = 32 

__C.TRAIN.NUM_WORKERS = 0

__C.TRAIN.MOMENTUM = 0.9

__C.TRAIN.WEIGHT_DECAY = 0.0001 #权值衰减

__C.TRAIN.CLS_WEIGHT = 1.0 

__C.TRAIN.LOC_WEIGHT = 1.0

__C.TRAIN.PRINT_FREQ = 20 #打印频次

__C.TRAIN.LOG_GRADS = False

__C.TRAIN.GRAD_CLIP = 10.0 #梯度裁剪

__C.TRAIN.BASE_LR = 0.005 #学习率

__C.TRAIN.LR = CN()

__C.TRAIN.LR.TYPE = 'log' #对数学习率衰减

__C.TRAIN.LR.KWARGS = CN(new_allowed=True)

__C.TRAIN.LR_WARMUP = CN()

__C.TRAIN.LR_WARMUP.WARMUP = True

__C.TRAIN.LR_WARMUP.TYPE = 'step' #递进学习率衰减，wormup

__C.TRAIN.LR_WARMUP.EPOCH = 5 #wormup代数

__C.TRAIN.LR_WARMUP.KWARGS = CN(new_allowed=True)

__C.MASK = CN()

__C.MASK.MASK = False 

# ------------------------------------------------------------------------ #
# Dataset options
# ------------------------------------------------------------------------ #
__C.DATASET = CN(new_allowed=True)

# Augmentation
# for template
__C.DATASET.TEMPLATE = CN()

# for detail discussion 数据增强
__C.DATASET.TEMPLATE.SHIFT = 4

__C.DATASET.TEMPLATE.SCALE = 0.05

__C.DATASET.TEMPLATE.BLUR = 0.0

__C.DATASET.TEMPLATE.FLIP = 0.0

__C.DATASET.TEMPLATE.COLOR = 1.0

__C.DATASET.SEARCH = CN()

__C.DATASET.SEARCH.SHIFT = 64

__C.DATASET.SEARCH.SCALE = 0.18

__C.DATASET.SEARCH.BLUR = 0.0

__C.DATASET.SEARCH.FLIP = 0.0

__C.DATASET.SEARCH.COLOR = 1.0

# for detail discussion 负样本概率
__C.DATASET.NEG = 0.2

# improve tracking performance for otb100
__C.DATASET.GRAY = 0.0

__C.DATASET.NAMES = ('VID', 'YOUTUBEBB', 'DET', 'COCO', 'GOT', 'LASOT')

__C.DATASET.VID = CN() 
__C.DATASET.VID.ROOT = ''          # VID dataset path
__C.DATASET.VID.ANNO = ''
__C.DATASET.VID.FRAME_RANGE = 100
__C.DATASET.VID.NUM_USE =   100000 #单数据集用量，不够就补齐

__C.DATASET.YOUTUBEBB = CN() 
__C.DATASET.YOUTUBEBB.ROOT = ''  #路径
__C.DATASET.YOUTUBEBB.ANNO = ''  #标签路径
__C.DATASET.YOUTUBEBB.FRAME_RANGE = 3 #采样范围，数据量大小
__C.DATASET.YOUTUBEBB.NUM_USE = 100000 #数据集用量，可以不全部使用，-1为全部，不足会自动填充

__C.DATASET.COCO = CN()
# __C.DATASET.COCO.ROOT = 'data/tank/crop511'
__C.DATASET.COCO.ROOT = 'C:\\Users\\li\\Desktop\\repo\\track\\SiamTrackers-master\\NanoTrack\\data\\results\\10GOT10K+10homemade\\crop511'
__C.DATASET.COCO.ANNO = 'C:\\Users\\li\\Desktop\\repo\\track\\SiamTrackers-master\\NanoTrack\\data\\results\\10GOT10K+10homemade\\crop511\\train.json'
__C.DATASET.COCO.FRAME_RANGE = 100
__C.DATASET.COCO.NUM_USE =  -1  

__C.DATASET.DET = CN()
__C.DATASET.DET.ROOT = ''
__C.DATASET.DET.ANNO = ''
__C.DATASET.DET.FRAME_RANGE = 1
__C.DATASET.DET.NUM_USE = 100000 

__C.DATASET.GOT = CN()
__C.DATASET.GOT.ROOT = 'data/GOT-10k/crop271'
__C.DATASET.GOT.ANNO = 'data/GOT-10k/crop271/train.json'
__C.DATASET.GOT.FRAME_RANGE = 100
__C.DATASET.GOT.NUM_USE = 100000

__C.DATASET.LASOT = CN()
__C.DATASET.LASOT.ROOT = ''
__C.DATASET.LASOT.ANNO = ''
__C.DATASET.LASOT.FRAME_RANGE = 100
__C.DATASET.LASOT.NUM_USE = 100000 

__C.DATASET.VIDEOS_PER_EPOCH = 600000  #每一代的总数据量，不足会自动填充
# ------------------------------------------------------------------------ #

# Backbone options
# ------------------------------------------------------------------------ #
__C.BACKBONE = CN()

__C.BACKBONE.TYPE = 'res50' #默认骨干是resnet50

__C.BACKBONE.KWARGS = CN(new_allowed=True)

# Pretrained backbone weights
__C.BACKBONE.PRETRAINED = ''

# Train layers
__C.BACKBONE.TRAIN_LAYERS = []

# Layer LR
__C.BACKBONE.LAYERS_LR = 0.1 #初始学习率

# Switch to train layer
__C.BACKBONE.TRAIN_EPOCH = 10 #骨干开始的代数

# ------------------------------------------------------------------------ #
# Adjust layer options
# ------------------------------------------------------------------------ #
__C.ADJUST = CN()

# Adjust layer
__C.ADJUST.ADJUST = True

__C.ADJUST.KWARGS = CN(new_allowed=True)

# Adjust layer type
__C.ADJUST.TYPE = "AdjustAllLayer"

# ------------------------------------------------------------------------ #
# BAN options
# ------------------------------------------------------------------------ #
__C.BAN = CN()

# Whether to use ban head
__C.BAN.BAN = False

# BAN type
__C.BAN.TYPE = 'MultiBAN' #原始输出头

__C.BAN.KWARGS = CN(new_allowed=True)

# ------------------------------------------------------------------------ #
# Point options
# ------------------------------------------------------------------------ #
__C.POINT = CN()

# Point stride
__C.POINT.STRIDE = 8 #粒子采样步长

# ------------------------------------------------------------------------ #
# Tracker options
# ------------------------------------------------------------------------ #
__C.TRACK = CN()

__C.TRACK.TYPE = 'NanoTracker'

# Scale penalty 惩罚系数
__C.TRACK.PENALTY_K = 0.16

# Window influence 窗口因子
__C.TRACK.WINDOW_INFLUENCE = 0.46

# Interpolation learning rate 学习率
__C.TRACK.LR = 0.34

# Exemplar size
__C.TRACK.EXEMPLAR_SIZE = 127

# Instance size
__C.TRACK.INSTANCE_SIZE = 255

# Base size
__C.TRACK.BASE_SIZE = 8

__C.TRACK.OUTPUT_SIZE = 16

# Context amount 
__C.TRACK.CONTEXT_AMOUNT = 0.5
