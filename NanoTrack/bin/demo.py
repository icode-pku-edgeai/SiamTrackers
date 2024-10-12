from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import time
import cv2
import torch
import numpy as np
from glob import glob

import sys 
sys.path.append(os.getcwd())  

from nanotrack.core.config import cfg
from nanotrack.models.model_builder import ModelBuilder
from nanotrack.tracker.tracker_builder import build_tracker
from nanotrack.utils.model_load import load_pretrain

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='tracking demo') 
parser.add_argument('--config', default='./models/config/configv3.yaml',type=str, help='config file')  #模型配置文件，覆盖nanotrack.core.config中的部分配置
# parser.add_argument('--snapshot', default='models/snapshot/test_20240919/checkpoint_e30.pth', type=str, help='model name')
parser.add_argument('--snapshot', default='models/pretrained/nanotrackv3.pth', type=str, help='model name')#预训练权重 
# parser.add_argument('--video_name', default=0, type=str, help='videos or image files')#视频加载目录
parser.add_argument('--video_name', default="./bin/35.mp4", type=str, help='videos or image files')
parser.add_argument('--save', action='store_true', help='whether visualzie result') #是否报名

args = parser.parse_args()

def get_frames(video_name): 
    if not video_name:
        cap = cv2.VideoCapture(0)  #摄像头采集
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break 

    elif video_name.endswith('avi') or \
        video_name.endswith('mp4') or \
        video_name.endswith('mov'):
        cap = cv2.VideoCapture(args.video_name) #视频读取
        
        # warmup
        for i in range(50):
            cap.read()

        while True:
            ret, frame = cap.read()
            if ret:
                yield frame 
            else:
                break
    else: #图片数据读取
        images = glob(os.path.join(video_name, '*.jpg*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame

def main():
    # load config
    cfg.merge_from_file(args.config) #nanotrack文件夹下的config和demo的config合并
    cfg.CUDA = torch.cuda.is_available() and cfg.CUDA  #调GPU
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder() #初始化模型

    # load model
    model = load_pretrain(model, args.snapshot).cuda().eval() #加载预训练权重到模型，并转cuda，转推理模式

    # build tracker
    tracker = build_tracker(model) #封装一个函数调用nanotrack初始化

    first_frame = True
    if args.video_name:#视频加载
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:#摄像头加载
        video_name = 'webcam'
    # cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)#创建窗口
    sum=0
    count=0    
    for frame in get_frames(args.video_name):
        
        if first_frame: #初帧
            # build video writer 
            if args.save: #保存视频时保存帧率
                if args.video_name.endswith('avi') or \
                    args.video_name.endswith('mp4') or \
                    args.video_name.endswith('mov'):
                    cap = cv2.VideoCapture(args.video_name)
                    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
                else:
                    fps = 30 #其他输入默认输出30fps
                #保存视频
                save_video_path = args.video_name.split(video_name)[0] + video_name + '_tracking.mp4'
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                frame_size = (frame.shape[1], frame.shape[0]) # (w, h)
                video_writer = cv2.VideoWriter(save_video_path, fourcc, fps, frame_size) 
            try:
                # init_rect = cv2.selectROI(video_name, frame, False, False)#手动选框
                # print(init_rect)
                init_rect = [244, 161, 74, 70]#默认位置

            except:
                exit()
            tracker.init(frame, init_rect)  #调用build_tracker中的nanotrack，进行单次初始化init，拿到首帧模板框的特征图
            first_frame = False
        else:
            count+=1
            t0 = time.time()#记录启动时间
            outputs = tracker.track(frame) #调用build_tracker中的nanotrack，调用跟踪
            sum+=float(outputs['best_score'])
            print('Done. (%.3fs)' % (time.time() - t0)) #输出推理总时间
            if 'polygon' in outputs: #多边形框输出，用来做分割任务
                polygon = np.array(outputs['polygon']).astype(np.int32)
                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                              True, (0, 255, 0), 3)
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)
            else:#仅输出矩形框
                bbox = list(map(int, outputs['bbox'])) #输出框并画图
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (0, 255, 0), 3)
                # 设置矩形位置（左上角）和大小  
                # rect_pt1 = (50, 50)  # 矩形左上角的坐标  
                # rect_pt2 = (200, 150)  # 矩形右下角的坐标  
                # text_position = (bbox[0] + (bbox[0] - bbox[0]) // 2, bbox[1] - 20)  # 根据矩形位置调整
                # text=str(outputs['best_score'])
                # cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # 白色文本，字体大小为0.7，线宽为2 
                # print(outputs['best_score'])
            # cv2.imshow(video_name, frame)
            cv2.waitKey(30) #等30ms，应该是防止摄像头采集速度过慢丢帧

        if args.save:
            video_writer.write(frame)#保存操作
    # print(sum/count)
    if args.save:
        video_writer.release()

if __name__ == '__main__':
    main()
