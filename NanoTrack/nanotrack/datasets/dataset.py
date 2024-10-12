# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import sys
import os

import cv2
import numpy as np
from torch.utils.data import Dataset

from nanotrack.utils.bbox import center2corner, Center
from nanotrack.datasets.point_target import PointTarget
from nanotrack.datasets.augmentation import Augmentation
from nanotrack.core.config import cfg

logger = logging.getLogger("global")

# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)


class SubDataset(object):
    def __init__(self, name, root, anno, frame_range, num_use, start_idx):
        cur_path = os.path.dirname(os.path.realpath(__file__))#获取当前的执行路径
        self.name = name #coco或者got
        self.root = os.path.join(cur_path, '../../', root)#指定数据根目录
        self.anno = os.path.join(cur_path, '../../', anno)#指定数据集标签路径
        self.frame_range = frame_range#前后帧选取范围，默认100
        self.num_use = num_use #使用量，-1为全部使用，初始为固定值
        self.start_idx = start_idx #开始位置索引，用于区分各个batch
        logger.info("loading " + name)
        with open(self.anno, 'r') as f:
            meta_data = json.load(f)#加载标签文件
            meta_data = self._filter_zero(meta_data)#数据清洗
        #洗掉000000中的6位0，命名非数字的内容，并排序
        for video in list(meta_data.keys()):#遍历视频
            for track in meta_data[video]:#遍历子目标
                frames = meta_data[video][track]#拿到帧数据
                frames = list(map(int,
                              filter(lambda x: x.isdigit(), frames.keys())))#将仅为数字的数据转为列表，去掉6位标识符
                frames.sort()#排序后返回
                meta_data[video][track]['frames'] = frames#记录int型的帧名称
                if len(frames) <= 0: #清洗空文件夹
                    logger.warning("{}/{} has no frames".format(video, track))
                    del meta_data[video][track]
        #清洗空文件夹
        for video in list(meta_data.keys()):
            if len(meta_data[video]) <= 0:
                logger.warning("{} has no tracks".format(video))
                del meta_data[video]

        self.labels = meta_data #清洗后的全部标签数据记录到类里面
        self.num = len(self.labels) #存储清洗后的视频量
        self.num_use = self.num if self.num_use == -1 else self.num_use #-1返回实际视频量、其他返回规定数据量
        self.videos = list(meta_data.keys())#视频名称
        logger.info("{} loaded".format(self.name))
        self.path_format = '{}.{}.{}.jpg'#路径格式
        self.pick = self.shuffle()#执行洗牌操作

    def _filter_zero(self, meta_data):#清洗标签
        meta_data_new = {}
        for video, tracks in meta_data.items():#遍历视频名称(文件夹名称)
            new_tracks = {}
            for trk, frames in tracks.items():#遍历(子目标名称)
                new_frames = {}
                for frm, bbox in frames.items():#遍历每一帧
                    if not isinstance(bbox, dict):
                        if len(bbox) == 4:#拿到正确的xyxy，并计算wh
                            x1, y1, x2, y2 = bbox
                            w, h = x2 - x1, y2 - y1
                        else:
                            w, h = bbox
                        if w <= 0 or h <= 0:
                            continue
                    new_frames[frm] = bbox
                if len(new_frames) > 0:
                    new_tracks[trk] = new_frames
            if len(new_tracks) > 0:
                meta_data_new[video] = new_tracks
        return meta_data_new#逐级保存

    def log(self):#打印数据集名称、子集起始索引、目标数据量、原始数据量、数据格式
        logger.info("{} start-index {} select [{}/{}] path_format {}".format(
            self.name, self.start_idx, self.num_use,
            self.num, self.path_format))

    def shuffle(self):
        lists = list(range(self.start_idx, self.start_idx + self.num))#初始化一个起始索引到实际的数据量的列表用于洗牌
        pick = []
        while len(pick) < self.num_use:#不断洗牌，直到满足规定用量
            np.random.shuffle(lists)
            pick += lists
        return pick[:self.num_use]#意味着实际量不足时，会重复洗牌进去

    def get_image_anno(self, video, track, frame):#获取路径与标签的键值对
        frame = "{:06d}".format(frame)#写死6位命名序号
        image_path = os.path.join(self.root, video,
                                  self.path_format.format(frame, track, 'x'))#拿到搜索图完整路径
        image_anno = self.labels[video][track][frame]#拿到标签信息，不区分x和z
        return image_path, image_anno

    def get_positive_pair(self, index):#正样本对，取一帧作为z，然后前后范围随机取一帧作为x，最后都取的是对应的x搜索图
        video_name = self.videos[index]#使用索引寻找视频名称
        video = self.labels[video_name]#使用视频名称找视频帧标签信息
        track = np.random.choice(list(video.keys()))#随机选一帧
        track_info = video[track]#获取随机帧的标签内容

        frames = track_info['frames']#取出随机帧的内容
        template_frame = np.random.randint(0, len(frames))#生成一个随机整数索引
        left = max(template_frame - self.frame_range, 0)#
        right = min(template_frame + self.frame_range, len(frames)-1) + 1
        search_range = frames[left:right]#实际就是不超出总帧数范围的前后frame_range随机取，作为搜索框，
        template_frame = frames[template_frame]#这一帧作为模板框
        search_frame = np.random.choice(search_range)#在收缩索引范围随机取一个值
        return self.get_image_anno(video_name, track, template_frame), \
            self.get_image_anno(video_name, track, search_frame)#拿到的都是图名称和标签

    def get_random_target(self, index=-1):#随机一个样本
        if index == -1:
            index = np.random.randint(0, self.num)#从已有数据量中随机取个值
        video_name = self.videos[index]#拿到视频名称
        video = self.labels[video_name]#拿到视频帧
        track = np.random.choice(list(video.keys()))#随机选一帧
        track_info = video[track]#拿到标签
        frames = track_info['frames']#拿到标签信息
        frame = np.random.choice(frames)#随机取一个目标
        return self.get_image_anno(video_name, track, frame)

    def __len__(self):
        return self.num 


class BANDataset(Dataset):
    def __init__(self,):
        super(BANDataset, self).__init__()

        # desired_size = (cfg.TRAIN.SEARCH_SIZE - cfg.TRAIN.EXEMPLAR_SIZE) / \
        #     cfg.POINT.STRIDE + 1 + cfg.TRAIN.BASE_SIZE
        # if desired_size != cfg.TRAIN.OUTPUT_SIZE:
        #     raise Exception('size not match!') 

        # create point target
        self.point_target = PointTarget()#粒子采样器，均匀撒点，随机分配图像内正负样本点

        # create sub dataset
        self.all_dataset = []#子集
        start = 0#数据起点
        self.num = 0#已用数据计数
        for name in cfg.DATASET.NAMES:#coco或者got
            subdata_cfg = getattr(cfg.DATASET, name)#拿到dataset的信息
            sub_dataset = SubDataset(#用来多数据集混用,数据量凑足目标值
                    name,
                    subdata_cfg.ROOT,#根目录
                    subdata_cfg.ANNO,#标签地址
                    subdata_cfg.FRAME_RANGE,#帧数，默认100
                    subdata_cfg.NUM_USE,#用量，-1就是全部都用,默认100000
                    start
                )
            start += sub_dataset.num#记录数据起点，用来遍历下个数据集
            self.num += sub_dataset.num_use#记录已用数据总数

            sub_dataset.log()#打印信息
            self.all_dataset.append(sub_dataset)#记录数据列表

        # data augmentation 模板和搜索框的数据增强
        self.template_aug = Augmentation(
                cfg.DATASET.TEMPLATE.SHIFT,#4
                cfg.DATASET.TEMPLATE.SCALE,#0.05
                cfg.DATASET.TEMPLATE.BLUR,#0
                cfg.DATASET.TEMPLATE.FLIP,#0
                cfg.DATASET.TEMPLATE.COLOR#1
            )
        self.search_aug = Augmentation(
                cfg.DATASET.SEARCH.SHIFT,#64
                cfg.DATASET.SEARCH.SCALE,#0.18
                cfg.DATASET.SEARCH.BLUR,#0.2
                cfg.DATASET.SEARCH.FLIP,#0
                cfg.DATASET.SEARCH.COLOR#1
            )
        videos_per_epoch = cfg.DATASET.VIDEOS_PER_EPOCH #每代设定的数据量，设定值400000
        self.num = videos_per_epoch if videos_per_epoch > 0 else self.num #设定值或实际量
        self.num *= cfg.TRAIN.EPOCH  #数据量*epoch量=2*10^7
        self.pick = self.shuffle()#洗牌

    def shuffle(self):
        pick = []
        m = 0
        while m < self.num:#洗到数据量满足为止
            p = []
            for sub_dataset in self.all_dataset:#逐个子数据集洗进去
                sub_p = sub_dataset.pick
                p += sub_p
            np.random.shuffle(p)
            pick += p
            m = len(pick)
        logger.info("shuffle done!")
        logger.info("dataset length {}".format(self.num))
        return pick[:self.num]

    def _find_dataset(self, index):
        for dataset in self.all_dataset:
            if dataset.start_idx + dataset.num > index:#数据索引index找数据属于哪个数据集的多少号
                return dataset, index - dataset.start_idx

    def _get_bbox(self, image, shape):#目标框转正方形模板框
        imh, imw = image.shape[:2]
        if len(shape) == 4:#xyxy转wh
            w, h = shape[2]-shape[0], shape[3]-shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE#127
        wc_z = w + context_amount * (w+h)
        hc_z = h + context_amount * (w+h)
        s_z = np.sqrt(wc_z * hc_z)#同面积正方形边长
        scale_z = exemplar_size / s_z#缩放比例
        w = w*scale_z
        h = h*scale_z
        cx, cy = imw//2, imh//2
        bbox = center2corner(Center(cx, cy, w, h))#中心点xywh转xyxy
        return bbox

    def __len__(self):
        return self.num

    def __getitem__(self, index):#索引访问
        index = self.pick[index]#索引内容
        dataset, index = self._find_dataset(index)#索引所在数据集

        gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()#默认为0，灰度图比例
        neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()#默认0.2，默认负样本比例

        # get one dataset
        if neg:
            template = dataset.get_random_target(index)#索引数据集随机取一个图
            search = np.random.choice(self.all_dataset).get_random_target()#所有数据集随机取个图
            # print('diff')
        else:
            template, search = dataset.get_positive_pair(index)#在索引集取正样本对
            # print('same')

        # get image 获取图片
        template_image = cv2.imread(template[0])
        search_image = cv2.imread(search[0])
        # print(template_image.shape)
        # print(search_image.shape)
        # get bounding box 获取边界框
        template_box = self._get_bbox(template_image, template[1])
        search_box = self._get_bbox(search_image, search[1])

        # augmentation 分别做数据增强
        template, _ = self.template_aug(template_image,
                                        template_box,
                                        cfg.TRAIN.EXEMPLAR_SIZE,
                                        gray=gray)

        search, bbox = self.search_aug(search_image,
                                       search_box,
                                       cfg.TRAIN.SEARCH_SIZE,
                                       gray=gray)

        # get labels 
        cls, delta = self.point_target(bbox, cfg.TRAIN.OUTPUT_SIZE, neg) #拿到分类和回归的目标值
        template = template.transpose((2, 0, 1)).astype(np.float32) #拿的都是x图？
        search = search.transpose((2, 0, 1)).astype(np.float32)#gbr转rbg转fp32
        # print(template.shape)
        # print(search.shape)
        return {
                'template': template,
                'search': search,
                'label_cls': cls,
                'label_loc': delta,
                'bbox': np.array(bbox)
                }
