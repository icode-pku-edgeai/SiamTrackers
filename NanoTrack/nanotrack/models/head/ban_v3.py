import torch
import torch.nn as nn
import math  

class BAN(nn.Module):#基类
    def __init__(self):
        super(BAN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

import torch.nn.functional as F

def xcorr_fast(x, kernel):
    """group conv2d to calculate cross correlation, fast version 使用卷积快速进行交叉相关的计算
    """
    batch = kernel.size()[0]
    pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
    px = x.view(1, -1, x.size()[2], x.size()[3])
    po = F.conv2d(px, pk, groups=batch)#忽略卷积的翻转操作计算交叉相关
    po = po.view(batch, -1, po.size()[2], po.size()[3]) 
    return po 

class UPChannelBAN(BAN):
    def __init__(self, feature_in=256, cls_out_channels=2):
        super(UPChannelBAN, self).__init__()

        cls_output = cls_out_channels
        loc_output = 4 
        
        self.template_cls_conv = nn.Conv2d(feature_in, 
                feature_in * cls_output, kernel_size=3)#模板图的分类卷积
        self.template_loc_conv = nn.Conv2d(feature_in, 
                feature_in * loc_output, kernel_size=3)#模板图的回归卷积

        self.search_cls_conv = nn.Conv2d(feature_in, 
                feature_in, kernel_size=3)#搜索图的分类卷积
        self.search_loc_conv = nn.Conv2d(feature_in, 
                feature_in, kernel_size=3)#搜索图的回归卷积

        self.loc_adjust = nn.Conv2d(loc_output, loc_output, kernel_size=1)

    def forward(self, z_f, x_f):
        cls_kernel = self.template_cls_conv(z_f)#模板图分类头
        loc_kernel = self.template_loc_conv(z_f)#模板图回归头

        cls_feature = self.search_cls_conv(x_f)#搜索图分类头
        loc_feature = self.search_loc_conv(x_f)#搜索图回归头

        cls = xcorr_fast(cls_feature, cls_kernel)#计算分类任务的模板和收缩的交叉相关性
        loc = self.loc_adjust(xcorr_fast(loc_feature, loc_kernel))#计算回归任务的模板和收缩的交叉相关性，然后转到指定box输出尺度
        return cls, loc

def xcorr_depthwise(x, kernel): 
    """depthwise cross correlation 深度可分离卷积计算搜索框和模板框的交叉相关性
    """
    batch = kernel.size(0)  #获取批次大小
    channel = kernel.size(1) #获取通道数
    x = x.view(1, batch*channel, x.size(2), x.size(3)) #重塑输入和卷积核
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))  
    out = F.conv2d(x, kernel,padding=1,groups=batch*channel)#深度可分离卷积计算相关性
    out = out.view(batch, channel, out.size(2), out.size(3)) #重塑输出

    return out

class DepthwiseXCorr(nn.Module): 
    def __init__(self, in_channels,  out_channels, kernel_size=3):
        super(DepthwiseXCorr, self).__init__()
        #模板和搜索框均使用逐点卷积
        self.conv_kernel = nn.Sequential(
                # pw 
                nn.Conv2d(in_channels,out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels) 
                )
        
        self.conv_search = nn.Sequential(
                # pw 
                nn.Conv2d(in_channels,out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
                ) 
        
        for modules in [self.conv_kernel, self.conv_search]:#赋值
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):#逐个卷积初始化为正态分布
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):#逐bn初始化w为1，b为0
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):#逐fc初始化w为正态分布，b为0
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_() 

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel) # 分别对模板框和搜索框进行卷积
        search = self.conv_search(search) # 1,96, 16, 16
        feature = xcorr_depthwise(search, kernel) #求相关性
        return feature 

class CAModule(nn.Module):
    """Channel attention module"""

    def __init__(self, channels=64, reduction=1):
        super(CAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):#平均池化-fc-relu-fc-sigmoid，得到CAM注意力
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

def xcorr_pixelwise(x,kernel): #z=kernel 矩阵乘法计算相关性
    """Pixel-wise correlation (implementation by matrix multiplication)
    The speed is faster because the computation is vectorized"""
    b, c, h, w = x.size() #获取输入尺寸
    kernel_mat = kernel.view((b, c, -1)).transpose(1, 2)  # (b, hz * wz, c) # 1,64,96 重塑卷积核
    x_mat = x.view((b, c, -1))  # (b, c, hx * wx) # 1, 96, 256 重塑输入特征图
    return torch.matmul(kernel_mat, x_mat).view((b, -1, h, w))  # (b, hz * wz, hx * wx) --> (b, hz * wz, hx, wx) 重塑输出

class PixelwiseXCorr(nn.Module): 
    def __init__(self, in_channels,  out_channels, kernel_size=3):
        super(PixelwiseXCorr, self).__init__()

        channels = 64

        self.CA_layer = CAModule(channels)#64通道的CAM注意力

        self.conv_kernel = nn.Sequential(
                #  pw 
                nn.Conv2d(in_channels,out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels) 
                )
        
        self.conv_search = nn.Sequential(
                # pw 
                nn.Conv2d(in_channels,out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
                ) 

        self.conv = nn.Sequential(
            # dw 深度卷积
            nn.Conv2d(channels, channels, kernel_size=2, groups=channels,bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU6(inplace=True),
            # pw 逐点卷积
            nn.Conv2d(channels, channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(channels),
        )

        for modules in [self.conv_kernel, self.conv_search, self.conv]:#赋予初值
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_() 

    def forward(self, kernel, search):  

        kernel=self.conv_kernel(kernel)#设置模板框和搜索框
        search=self.conv_search(search)
        
        feature = xcorr_pixelwise(search,kernel) #矩阵乘法计算模板框和收缩框的相关性
       
        corr = self.CA_layer(feature) #使用cam注意力机制
        
        corr=self.conv(corr) #卷积输出

        return corr

class DepthwiseBAN(BAN): 
    def __init__(self, in_channels=96, out_channels=96, weighted=False):
        super(DepthwiseBAN, self).__init__()
        #回归
        self.corr_dw_reg = DepthwiseXCorr(in_channels, out_channels)#深度卷积、回归相关性
        self.corr_pw_reg = PixelwiseXCorr(in_channels, out_channels)#矩阵乘法、回归相关性
        #分类
        self.corr_dw_cls = DepthwiseXCorr(in_channels, out_channels)#深度卷积、分类相关性
        self.corr_pw_cls = PixelwiseXCorr(in_channels, out_channels)#矩阵乘法、分类相关性
        
        cls_tower = []
        bbox_tower = [] 
        
        #------------------------------------------------------cls-----------------------------------------------------#
        for i in range(6):  # 定义分类塔，6个(3*3深度卷积-bn-relu6-1*1卷积-bn)
            # dw
            cls_tower.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False))
            cls_tower.append(nn.BatchNorm2d(in_channels))
            cls_tower.append(nn.ReLU6(inplace=True))

            # pw-linear 
            cls_tower.append(nn.Conv2d(in_channels,in_channels, kernel_size=1, stride=1, padding=0, bias=False))
            cls_tower.append(nn.BatchNorm2d(in_channels))
        
        #------------------------------------------------------box-----------------------------------------------------#
        for i in range(6): #定义回归塔，6个(3*3深度卷积-bn-relu6-1*1卷积-bn)
            # dw 
            bbox_tower.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False))
            bbox_tower.append(nn.BatchNorm2d(in_channels))
            bbox_tower.append(nn.ReLU6(inplace=True))

            # pw-linear  
            bbox_tower.append(nn.Conv2d(in_channels,in_channels, kernel_size=1, stride=1, padding=0, bias=False))
            bbox_tower.append(nn.BatchNorm2d(in_channels))

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))

        self.cls_logits =nn.Sequential(
            nn.Conv2d(in_channels,  2, kernel_size=1, stride=1, padding=0),    #回归输出2，前景背景
        ) 
        
        self.bbox_pred =nn.Sequential( 
            nn.Conv2d(in_channels, 4, kernel_size=1, stride=1, padding=0),  #分类输出4，xyxy
        )

        self.down_reg = nn.Sequential( 
            nn.Conv2d(in_channels+64, in_channels, kernel_size=1, stride=1, padding=0),#回归降采样
        )

        self.down_cls = nn.Sequential( 
            nn.Conv2d(in_channels+64, in_channels, kernel_size=1, stride=1, padding=0),#分类降采样
        )

        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,self.down_cls,self.down_reg]:#赋值
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1) 
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear): 
                    m.weight.data.normal_(0, 0.01) 
                    m.bias.data.zero_()

    def crop(self,x): #特征图裁剪
        if x.size(3) >4: 
            l = 2
            r = l + 4 
            x = x[:, :, l:r, l:r]
        return x 

    def forward(self, z_f, x_f): #训练使用

        crop_z_f=self.crop(z_f)#模板特征图裁剪
       
        x_pw_reg = self.corr_pw_reg(z_f, x_f) #原模板图和搜索图、矩阵乘法、回归相关性
        x_pw_cls = self.corr_pw_cls(z_f, x_f) # 原模板图和搜索图、矩阵乘法、分类相关性
        
        x_dw_reg = self.corr_dw_reg(crop_z_f, x_f)#裁剪的模板图和搜索图、深度卷积、回归相关性
        x_dw_cls = self.corr_dw_cls(crop_z_f, x_f) #裁剪的模板图和搜索图、深度卷积、分类相关性

        x_reg = self.down_reg(torch.cat((x_pw_reg,x_dw_reg), 1))# 深度卷积和矩阵乘法计算的回归相关性cancat，降采样
        x_cls = self.down_cls(torch.cat((x_pw_cls,x_dw_cls), 1))# 深度卷积和矩阵乘法计算的分类相关性cancat，降采样
        
        cls_tower = self.cls_tower(x_cls)#过分类塔
        logits = self.cls_logits(cls_tower)#生成分类输出

        bbox_tower=self.bbox_tower(x_reg)#过回归塔
        bbox_reg=self.bbox_pred(bbox_tower)#生成回归输出
        bbox_reg = torch.exp(bbox_reg)#取对数
        
        return logits, bbox_reg
