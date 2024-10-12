__author__ = 'tylin'
__version__ = '2.0'
 
import json
import time
 
try:
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon
except Exception as e:
    print(e)
import numpy as np
import copy
import itertools
import os
from collections import defaultdict
import sys
PYTHON_VERSION = sys.version_info[0]
if PYTHON_VERSION == 2:
    from urllib import urlretrieve
elif PYTHON_VERSION == 3:
    from urllib.request import urlretrieve
 
def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')
 
class COCO:
    def __init__(self, annotation_file=None):  # ./annotations/instances_val2017.json  ./annotations/instances_train2017.json
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        # load dataset
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)  # 利用 defaultdict(list) 构建 键值对应列表
        if not annotation_file == None:  # 传入的是 注释的json文件
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))  # <c>
            assert type(dataset)==dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time()- tic))
            self.dataset = dataset  # 上面已经加载了 json 文件
            self.createIndex()
 
    def createIndex(self):
        # create index
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        imgToAnns,catToImgs = defaultdict(list),defaultdict(list)  # imgToAnns: 图片id对应注释  catToImgs: 一个类别id对应的所有该类别下的图片id
        if 'annotations' in self.dataset:  #
            for ann in self.dataset['annotations']:  # <c>
                imgToAnns[ann['image_id']].append(ann)
                anns[ann['id']] = ann
 
        if 'images' in self.dataset:
            for img in self.dataset['images']:  # 包含图片的一些信息， 路径，长宽等
                imgs[img['id']] = img
 
        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:  # 类别信息
                cats[cat['id']] = cat
 
        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToImgs[ann['category_id']].append(ann['image_id'])
 
        print('index created!')
 
        # create class members
        self.anns = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats
 
    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset['info'].items():
            print('{}: {}'.format(key, value))
 
    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]  # 后面的成立
        catIds = catIds if _isArrayLike(catIds) else [catIds]  # catIds={list:0} []
 
        if len(imgIds) == len(catIds) == len(areaRng) == 0:  # False
            anns = self.dataset['annotations']
        else:
            if not len(imgIds) == 0:  # True
                lists = [self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns]
                anns = list(itertools.chain.from_iterable(lists))  # 迭代拿出其中的元素装进一个列表里 <c>
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds)  == 0 else [ann for ann in anns if ann['category_id'] in catIds]  # <c> 1
            anns = anns if len(areaRng) == 0 else [ann for ann in anns if ann['area'] > areaRng[0] and ann['area'] < areaRng[1]]  # <c> 2 areaRng 传入的参数， 这里为 []
        if not iscrowd == None:  # False
            ids = [ann['id'] for ann in anns if ann['iscrowd'] == iscrowd]
        else:
            ids = [ann['id'] for ann in anns]  # <c> anns中 字典的 键id 对应的值  <c>
        return ids
 
    def loadImgs(self, ids=[]):  # 图片的信息，不是具体的图片矩阵
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if _isArrayLike(ids):
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:  # True
            # a = self.imgs[ids]  # 包含图片的路径啥的信息， <c>
            return [self.imgs[ids]]
 
    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):  # True
            # b = [self.anns[id] for id in ids]  # <c>
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]