# import os
# import cv2
# import json
# import logging
# import os.path as osp
# from tqdm import tqdm
# from functools import partial
# from multiprocessing import Pool, cpu_count

# def set_logging(name=None):
#     rank = int(os.getenv('RANK', -1))
#     logging.basicConfig(format="%(message)s", level=logging.INFO if (rank in (-1, 0)) else logging.WARNING)
#     return logging.getLogger(name)

# LOGGER = set_logging(__name__)

# def process_img(image_filename, data_path, label_path):
#     # Open the image file to get its size
#     image_path = os.path.join(data_path, image_filename)
#     img = cv2.imread(image_path)
#     height, width = img.shape[:2]

#     # Open the corresponding label file
#     label_file = os.path.join(label_path, os.path.splitext(image_filename)[0] + ".txt")
#     with open(label_file, "r") as file:
#         lines = file.readlines()

#     # Process the labels
#     labels = []
#     for line in lines:
#         category, x, y, w, h = map(float, line.strip().split())
#         labels.append((category, x, y, w, h))

#     return image_filename, {"shape": (height, width), "labels": labels}

# def get_img_info(data_path, label_path):
#     LOGGER.info(f"Get img info")

#     image_filenames = os.listdir(data_path)

#     with Pool(cpu_count()) as p:
#         results = list(tqdm(p.imap(partial(process_img, data_path=data_path, label_path=label_path), image_filenames), total=len(image_filenames)))

#     img_info = {image_filename: info for image_filename, info in results}
#     return img_info


# def generate_coco_format_labels(img_info, class_names, save_path):
#     # for evaluation with pycocotools
#     dataset = {"categories": [], "annotations": [], "images": []}
#     for i, class_name in enumerate(class_names):
#         dataset["categories"].append(
#             {"id": i, "name": class_name, "supercategory": ""}
#         )

#     ann_id = 0
#     LOGGER.info(f"Convert to COCO format")
#     for i, (img_path, info) in enumerate(tqdm(img_info.items())):
#         labels = info["labels"] if info["labels"] else []
#         img_id = osp.splitext(osp.basename(img_path))[0]
#         img_h, img_w = info["shape"]
#         dataset["images"].append(
#             {
#                 "file_name": os.path.basename(img_path),
#                 "id": img_id,
#                 "width": img_w,
#                 "height": img_h,
#             }
#         )
#         if labels:
#             for label in labels:
#                 c, x, y, w, h = label[:5]
#                 # convert x,y,w,h to x1,y1,x2,y2
#                 x1 = (x - w / 2) * img_w
#                 y1 = (y - h / 2) * img_h
#                 x2 = (x + w / 2) * img_w
#                 y2 = (y + h / 2) * img_h
#                 # cls_id starts from 0
#                 cls_id = int(c)
#                 w = max(0, x2 - x1)
#                 h = max(0, y2 - y1)
#                 dataset["annotations"].append(
#                     {
#                         "area": h * w,
#                         "bbox": [x1, y1, w, h],
#                         "category_id": cls_id,
#                         "id": ann_id,
#                         "image_id": img_id,
#                         "iscrowd": 0,
#                         # mask
#                         "segmentation": [],
#                     }
#                 )
#                 ann_id += 1

#     with open(save_path, "w") as f:
#         json.dump(dataset, f)
#         LOGGER.info(
#             f"Convert to COCO format finished. Resutls saved in {save_path}"
#         )


# if __name__ == "__main__":
    
#     # Define the paths
#     data_path   = "C:\\Users\\li\\Desktop\\repo\\SiamTrackers-master\\NanoTrack\\data\\coco128\\images\\train"
#     label_path  = "C:\\Users\\li\\Desktop\\repo\\SiamTrackers-master\\NanoTrack\\data\\coco128\\labels\\train"

#     # class_names = ["tank"]
#     class_names = [ "person", "bicycle", "car", "motorcycle", "airplane",
#     "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
#     "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
#     "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
#     "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
#     "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
#     "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
#     "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
#     "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
#     "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
#     "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
#     "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
#     "scissors", "teddy bear", "hair drier", "toothbrush"]  # 类别名称请务必与 YOLO 格式的标签对应
#     save_path   = "C:\\Users\\li\\Desktop\\repo\\SiamTrackers-master\\NanoTrack\\data\\coco128\\annotations\\train2017.json"

#     img_info = get_img_info(data_path, label_path)
#     generate_coco_format_labels(img_info, class_names, save_path)


import os
import json
import cv2
import random
import time
from PIL import Image

coco_format_save_path='C:\\Users\\li\\Desktop\\repo\\SiamTrackers-master\\NanoTrack\\data\\tank\\annotations\\'   #要生成的标准coco格式标签所在文件夹
yolo_format_classes_path='C:\\Users\\li\\Desktop\\repo\\SiamTrackers-master\\NanoTrack\\data\\tank\\classes.txt'     #类别文件，一行一个类
yolo_format_annotation_path='C:\\Users\\li\\Desktop\\repo\\SiamTrackers-master\\NanoTrack\\data\\tank\\labels\\val\\'  #yolo格式标签所在文件夹
img_pathDir='C:\\Users\\li\\Desktop\\repo\\SiamTrackers-master\\NanoTrack\\data\\tank\\images\\val\\'    #图片所在文件夹

with open(yolo_format_classes_path,'r') as fr:                               #打开并读取类别文件
    lines1=fr.readlines()
# print(lines1)
categories=[]                                                                 #存储类别的列表
for j,label in enumerate(lines1):
    label=label.strip()
    categories.append({'id':j+1,'name':label,'supercategory':'None'})         #将类别信息添加到categories中
# print(categories)

write_json_context=dict()                                                      #写入.json文件的大字典
write_json_context['info']= {'description': '', 'url': '', 'version': '', 'year': 2024, 'contributor': '纯粹ss', 'date_created': '2024-01-12'}
write_json_context['licenses']=[{'id':1,'name':None,'url':None}]
write_json_context['categories']=categories
write_json_context['images']=[]
write_json_context['annotations']=[]

#接下来的代码主要添加'images'和'annotations'的key值
imageFileList=os.listdir(img_pathDir)                                           #遍历该文件夹下的所有文件，并将所有文件名添加到列表中
for i,imageFile in enumerate(imageFileList):
    imagePath = os.path.join(img_pathDir,imageFile)                             #获取图片的绝对路径
    image = Image.open(imagePath)                                               #读取图片，然后获取图片的宽和高
    W, H = image.size

    img_context={}                                                              #使用一个字典存储该图片信息
    #img_name=os.path.basename(imagePath)                                       #返回path最后的文件名。如果path以/或\结尾，那么就会返回空值
    img_context['file_name']=imageFile
    img_context['height']=H
    img_context['width']=W
    img_context['date_captured']='2024.1.12'
    img_context['id']=i                                                         #该图片的id
    img_context['license']=1
    img_context['color_url']=''
    img_context['flickr_url']=''
    write_json_context['images'].append(img_context)                            #将该图片信息添加到'image'列表中


    txtFile=imageFile[:6]+'.txt'                                               #获取该图片获取的txt文件
    with open(os.path.join(yolo_format_annotation_path,txtFile),'r') as fr:
        lines=fr.readlines()                                                   #读取txt文件的每一行数据，lines2是一个列表，包含了一个图片的所有标注信息
    for j,line in enumerate(lines):

        bbox_dict = {}                                                          #将每一个bounding box信息存储在该字典中
        # line = line.strip().split()
        # print(line.strip().split(' '))

        class_id,x,y,w,h=line.strip().split(' ')                                          #获取每一个标注框的详细信息
        class_id,x, y, w, h = int(class_id), float(x), float(y), float(w), float(h)       #将字符串类型转为可计算的int和float类型

        xmin=(x-w/2)*W                                                                    #坐标转换
        ymin=(y-h/2)*H
        xmax=(x+w/2)*W
        ymax=(y+h/2)*H
        w=w*W
        h=h*H

        bbox_dict['id']=i*10000+j                                                         #bounding box的坐标信息
        bbox_dict['image_id']=i
        bbox_dict['category_id']=class_id+1                                               #注意目标类别要加一
        bbox_dict['iscrowd']=0
        height,width=abs(ymax-ymin),abs(xmax-xmin)
        bbox_dict['area']=height*width
        bbox_dict['bbox']=[xmin,ymin,w,h]
        bbox_dict['segmentation']=[[xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax]]
        write_json_context['annotations'].append(bbox_dict)                               #将每一个由字典存储的bounding box信息添加到'annotations'列表中

name = os.path.join(coco_format_save_path,"val"+ '.json')
with open(name,'w') as fw:                                                                #将字典信息写入.json文件中
    json.dump(write_json_context,fw,indent=2)


