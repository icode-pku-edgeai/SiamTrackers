# -*- coding:utf8 -*-
from PIL import Image
import os
import glob
import numpy as np
 
def txt2det(yolo_labels_dir, mot_labels_det_path):
    object_id = 1  # 目标ID
    with open(mot_labels_det_path, 'w') as mot_file:
        for file in os.listdir(yolo_labels_dir):
            if file.endswith(".txt"):
                yolo_file_path = os.path.join(yolo_labels_dir, file)
                frame_idx = int(file.split("_")[-1].split(".")[0])  # 提取帧索引
                image_width = 640
                image_height=480
 
                with open(yolo_file_path, 'r') as yolo_file:
                    lines = yolo_file.readlines()
 
                    for line in lines:
                        data = line.split()
                        class_id = int(data[0])
                        x_center = float(data[1])
                        y_center = float(data[2])
                        width = float(data[3])
                        height = float(data[4])
 
                        left = int((x_center - width / 2) * image_width)
                        top = int((y_center - height / 2) * image_height)
                        # right = int((x_center + width / 2) * image_width)
                        # bottom = int((y_center + height / 2) * image_height)
                        w = int(width*image_width)
                        h = int(height * image_height)
 
                        # 写入MOT标签文件
                        mot_file.write("{},-1,{},{},{},{},1,-1,-1,-1\n".format(frame_idx, left, top, w, h))
                object_id += 1
 
def txt2gt(yolo_labels_dir, mot_labels_gt_path):
    object_id = 1  # 目标ID
    with open(mot_labels_gt_path, 'w') as mot_file:
        for file in os.listdir(yolo_labels_dir):
            if file.endswith(".txt"):
                yolo_file_path = os.path.join(yolo_labels_dir, file)
                frame_idx = int(file.split("_")[-1].split(".")[0])  # 提取帧索引
                image_width = 640
                image_height=480
 
                with open(yolo_file_path, 'r') as yolo_file:
                    lines = yolo_file.readlines()
 
                    for line in lines:
                        data = line.split()
                        class_id = int(data[0])
                        x_center = float(data[1])
                        y_center = float(data[2])
                        width = float(data[3])
                        height = float(data[4])
 
                        left = int((x_center - width / 2) * image_width)
                        top = int((y_center - height / 2) * image_height)
                        # right = int((x_center + width / 2) * image_width)
                        # bottom = int((y_center + height / 2) * image_height)
                        w = int(width*image_width)
                        h = int(height * image_height)
 
                        # 写入MOT标签文件
                        mot_file.write("{},{},{},{},{},{},0,{},1\n".format(frame_idx,object_id,left, top, w, h,class_id))
                object_id += 1
 
def main():
    # 设置YOLO标签文件夹和MOT标签文件路径
    yolo_labels_dir = "/labels/val/"
    mot_labels_det_path = "/labels/det/mot_det.txt"
    mot_labels_gt_path = "/labels/gt/mot_gt.txt"
    # 调用转换函数
    txt2det(yolo_labels_dir, mot_labels_det_path)
    txt2gt(yolo_labels_dir, mot_labels_gt_path)
 
if __name__ == '__main__':
    main()
 
 