# got-10k数据集处理
## 原数据格式
```
GOT-10K
├── train
│   ├── GOT-10K_Train_000001:视频连续帧，单目标
│   │     ├── 00000001.jpg
│   │     ├── 00000002.jpg
│   │     ├── ......
│   │     └── groundtruth.txt：左上角xy+wh值
│   ├── GOT-10K_Train_000002
│   ├── GOT-10K_Train_000003
│   ├── ......
│   └── list.txt:标签类别名称
├── val
└── test
```
## parse_got10k.py
+ 提供数据集路径，路径下包括train和val文件夹，任一文件夹底下包括多个数据文件夹，任一数据文件夹下包括一个groundtruth文本文件，图像文件名从00000001开始
+ 运行生成got10k_calib.json
## par_crop.py
+ 提供数据集路径，路径下包括train和val文件夹，任一文件夹底下包括多个数据文件夹，任一数据文件夹下包括一个groundtruth文本文件，图像文件名从00000001开始
+ 有个np的bug，需要将np.float改为np.float64
+ 输出地址在main函数中指出
+ main函数可设置裁切尺寸和线程数，考虑模型输入为255，数据建议裁剪为255或511（CSDN推荐），271（nanotrack）作者推荐，模板图片固定为127尺寸
+ 输出：原图片文件夹下00000001.jpg将输出为000000.00.x.jpg和000000.00.z.jpg
## gen_json.py
+ 加载第一个步骤生成的got10k_calib.json，指定位置分别输出train.json和val.json

# coco数据集处理
**coco格式的数据训练效果并不好**
+ coco数据集不是连续视频帧组成文件夹，而且是多目标，单个txt或总的json格式而不是总的groundtruth
+ coco的数据格式为中心点的xywh，要转成左上角点的xywh
## split_datasets.py
+ 划分数据集为train和val，
## rename.py
+ 从000000开始重命名
## txt2json.py
+ 从txt格式的标签中生成json格式的标签到
+ 在annotation文件夹下生成train.json和val.json
+ 使用test_json.py测试json文件的正确性
## par_crop.py和coco.py
+ np.float的bug仍旧存在，改为np.float64
+ 注意输入输出地址在Windows下的格式、裁剪尺寸
+ 会将每一幅图像的每一个目标输出成511和127两个尺寸的图片
## gen_json.py
+ 修改好annotation的地址和名称，导出train.json和val.json

# 训练
## 数据准备
+ 数据准备在nanotrack目录下的data文件夹中
## 主要训练文件 bin/train.py
### models/config/configv3.yaml
#### 骨干网络
+ 类型：mobilenetv3的small版本
+ 层数:4层
+ 预训练权重：models/pretrained
+ 训练代数：10
+ 学习率：0.1
#### 特征图调整网络
+ 名称：AdjustLayer
+ 输入输出通道数：96
#### 深度可分离的平衡注意力网络
+ 名称：DepthwiseBAN
+ 输入输出通道数：96
#### 追踪网络
+ 名称：
+ 窗口影响因子
+ 惩罚系数
+ 学习率
+ 图像尺寸
#### 训练参数
+ 代数
+ batch值
+ 线程数
+ 分类与回归损失权重
+ 存储位置
+ 学习率
#### 数据集
+ 名称
+ 地址
## nanotrack/core/config.py
+ workers
+ 数据集地址，例如coco.anno
+ 