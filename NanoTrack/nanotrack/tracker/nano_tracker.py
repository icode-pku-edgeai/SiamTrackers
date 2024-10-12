from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from nanotrack.core.config import cfg
from nanotrack.tracker.base_tracker import SiameseTracker
from nanotrack.utils.bbox import corner2center

class NanoTracker(SiameseTracker):
    def __init__(self, model):
        super(NanoTracker, self).__init__()
        # self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
        #     cfg.POINT.STRIDE + 1 + cfg.TRACK.BASE_SIZE

        self.score_size =cfg.TRACK.OUTPUT_SIZE #输出尺度 15，,15*15=225

        hanning = np.hanning(self.score_size) #长度为15的汉宁窗数组，系数由余弦的半周期构成
        window = np.outer(hanning, hanning)
        self.cls_out_channels = 2 #分类输出通道
        self.window = window.flatten()#225长的一个定值标量
        
        self.points = self.generate_points(cfg.POINT.STRIDE, self.score_size)#225，2的二维矩阵，225就是输出的结果数，其实就是
        self.model = model
        self.model.eval()

    def generate_points(self, stride, size):#均匀撒点
        ori = - (size // 2) * stride
        x, y = np.meshgrid([ori + stride * dx for dx in np.arange(0, size)],
                           [ori + stride * dy for dy in np.arange(0, size)])
        points = np.zeros((size * size, 2), dtype=np.float32)
        points[:, 0], points[:, 1] = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()

        return points

    def _convert_bbox(self, delta, point):#预测的偏差量转实际xyxy，再转xywh
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.detach().cpu().numpy()
        #预测的偏差添加到xyxy中
        delta[0, :] = point[:, 0] - delta[0, :] #x1
        delta[1, :] = point[:, 1] - delta[1, :] #y1
        delta[2, :] = point[:, 0] + delta[2, :] #x2
        delta[3, :] = point[:, 1] + delta[3, :] #y2
        delta[0, :], delta[1, :], delta[2, :], delta[3, :] = corner2center(delta)#xyxy转中心点xywh
        return delta

    def _convert_score(self, score): #分类输出通道，写死2，应该对应是255和127
        if self.cls_out_channels == 1:
            score = score.permute(1, 2, 3, 0).contiguous().view(-1)
            score = score.sigmoid().detach().cpu().numpy()
        else:
            score = score.permute(1, 2, 3, 0).contiguous().view(self.cls_out_channels, -1).permute(1, 0)
            score = score.softmax(1).detach()[:, 1].cpu().numpy()#实际就是softmax转0-1范围
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary): #
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox): #初始化函数，用于首帧调用，例如bbox为(516, 129, 231, 382)
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox 
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])#拿出中心点，例如（631,319.5）
        self.size = np.array([bbox[2], bbox[3]]) #目标框的bbox转xy和wh，注意前后帧传递的信息就在这个self.size里面，例如（231, 382）

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)#(w+h)/2+w
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)#(w+h)/2+h
        s_z = round(np.sqrt(w_z * h_z)) #将矩形目标框的wh转为正方形，例如608

        # calculate channle average  计算原图在RGB三通道上的平均值
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop 使用原图、中心点、模板宽高、原图宽高、通道上的平均值 得到相同面积的矩形模板图
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.model.template(z_crop) #调用ModelBuilder中的template，模板图单独过骨干网络

    def track(self, img):
        """
        args: 
            img(np.ndarray): BGR image 
        return:
            bbox(list):[x, y, width, height]  
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z) #将上一帧预测的矩形目标框的wh转为正方形
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z #模板127/目标框实际尺寸得到模板缩放比例
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE) #上一帧的矩形目标框等比放大到搜索框尺寸，比例为搜索框255/模板框127
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)#圆整后裁出目标的搜索框，实际就是用上一帧预测下一帧

        outputs = self.model.track(x_crop)#调用ModelBuilder中的track，预测的搜索框过骨干得到特征图，然后与头帧的模板框特征图一起过头部，得到分类和回归的输出

        score = self._convert_score(outputs['cls'])#计算一组分类置信度值，225个结果，绝大多数为10-5次方，某区域集中在0.999
        pred_bbox = self._convert_bbox(outputs['loc'], self.points)#计算一组bbox的xywh，对应225*4
        
        def change(r):
            return np.maximum(r, 1. / r)#求二者的逐元素最大值

        def sz(w, h): 
            pad = (w + h) * 0.5 #求wh的均值
            return np.sqrt((w + pad) * (h + pad)) #加权wh后求转正方形
        
        # scale penalty 预测框转正方形/前帧结果缩放，逐元素找最大值作为缩放惩罚尺度，其实就是找box面积变化最小的，集中在1-1.3之间
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty (上帧w/h)/(预测框w/h)，逐元素找最大值作为比例惩罚尺度，其实就是找box宽高比变化最小的，集中在1-8之间
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K) #由上述二者计算一个惩罚值，在0-1之间

        # score 这个惩罚系数就是用来评估框的变化大小，取代nms做多框筛选
        pscore = penalty * score #加权置信度分数

        # window penalty 加权窗口惩罚系数(都是定值)，这个系数v123版本是不同的，窗口惩罚实际上是因为图像边缘框的检测不具有线性的原因
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE

        best_idx = np.argmax(pscore) #找最大值的索引

        bbox = pred_bbox[:, best_idx] / scale_z #找最好的box框

        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR #最优解下的惩罚值*分数*学习率，此处的学习率应该是前后帧关系的学习率，而不是训练的学习率

        cx = bbox[0] + self.center_pos[0]#中心点xy

        cy = bbox[1] + self.center_pos[1]  

        # smooth bbox 最后的wh结果是由前帧和预测值移动加权平均得到的
        width = self.size[0] * (1 - lr) + bbox[2] * lr 

        height = self.size[1] * (1 - lr) + bbox[3] * lr 

        # clip boundary 超边界判断
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])

        # udpate state 更新中心点和wh保留到下一帧
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height] #转左上角的xywh

        best_score = score[best_idx]#记录最好分数
        return {
                'bbox': bbox,
                'best_score': best_score
               }
