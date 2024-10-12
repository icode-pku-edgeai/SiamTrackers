from os import mkdir, makedirs
from os.path import join, isdir
from concurrent import futures
import cv2
import sys
import time
import numpy as np
from coco import COCO
 
def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()
 
def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz-1) / (bbox[2]-bbox[0])  # ratio size / w
    b = (out_sz-1) / (bbox[3]-bbox[1])  # ratio size / h
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float64)  # 变换矩阵，这个应该就是实现以目标为中心，然后先以填充裁剪再resize
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)  # 利用仿射变换进行裁剪
    return crop
 
def pos_s_2_bbox(pos, s):  # 以目标为中心，返回填充后区域的左上角和右下角坐标 （x1,y1,x2,y2）
    return [pos[0]-s/2, pos[1]-s/2, pos[0]+s/2, pos[1]+s/2]
 
def crop_like_SiamFC(image, bbox, context_amount=0.5, exemplar_size=127, instanc_size=255, padding=(0, 0, 0)):
    target_pos = [(bbox[2]+bbox[0])/2., (bbox[3]+bbox[1])/2.]  # [(x1+x2)/2, (y1+y2)/2] bbox的中心坐标，等价于 x1+(x2-x1)/2
    target_size = [bbox[2]-bbox[0], bbox[3]-bbox[1]]  # w, h
    wc_z = target_size[1] + context_amount * sum(target_size)  # w+(w+h)/2
    hc_z = target_size[0] + context_amount * sum(target_size)  # h+(w+h)/2
    s_z = np.sqrt(wc_z * hc_z)  # [(w+(w+h)/2)(h+(w+h)/2)]^1/2
    scale_z = exemplar_size / s_z  # ratio
    d_search = (instanc_size - exemplar_size) / 2  # 尺寸差值，单边到单边
    pad = d_search / scale_z  # 填充时的一边范围的差值
    s_x = s_z + 2 * pad  # 直接加上填充的差值，就等于x填充后的尺寸
 
    z = crop_hwc(image, pos_s_2_bbox(target_pos, s_z), exemplar_size, padding)  # 返回裁剪后的模板图片
    x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instanc_size, padding)  # 搜索图片
    return z, x
 
def crop_img(img, anns, set_crop_base_path, set_img_base_path, instanc_size=511):  # img,anns: <c> set_crop_base_path:'./crop511/val2017', set_img_base_path:'./val2017'
    frame_crop_base_path = join(set_crop_base_path, img['file_name'].split('/')[-1].split('.')[0])  # 举例 './crop511/val2017/000000386912'
    if not isdir(frame_crop_base_path): makedirs(frame_crop_base_path)
 
    im = cv2.imread('{}/{}'.format(set_img_base_path, img['file_name']))  # 读入图片 举例 (480,640,3) 路径:'./val2017/000000386912.jpg'
    avg_chans = np.mean(im, axis=(0, 1))  #  用于裁剪的填充 举例 [97.14272461, 99.71438477, 105.41124349]
    for trackid, ann in enumerate(anns):  # ann <c>
        rect = ann['bbox']  # ground_truth bbox 举例 [210.27,143.29,219.82,276.15]
        bbox = [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]  # xywh ---> xyxy
        if rect[2] <= 0 or rect[3] <=0:
            continue
        z, x = crop_like_SiamFC(im, bbox, instanc_size=instanc_size, padding=avg_chans)
        cv2.imwrite(join(frame_crop_base_path, '{:06d}.{:02d}.z.jpg'.format(0, trackid)), z)  # 保存裁剪后的图片
        cv2.imwrite(join(frame_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(0, trackid)), x)  # 这里裁剪后的x图片的尺寸为511
 
def main(instanc_size=511, num_threads=12):
    dataDir = 'C:\\Users\\li\\Desktop\\repo\\SiamTrackers-master\\NanoTrack\\data\\origin_dataset\\homemade_video'
    crop_path = '{}\\crop_temp{:d}'.format(dataDir,instanc_size)  # 路径 ./crop511
    if not isdir(crop_path): mkdir(crop_path)  # 如果不存在该目录，则创建该目录
 
    for dataType in ['val_temp', 'train_temp']:  # 传入的两个数据集的目录
        set_crop_base_path = join(crop_path, dataType)  # 创建路径， ./crop511/val2017   ./crop511/train2017
        set_img_base_path = '{}\\{}'.format(dataDir,dataType) #  ./val2017   ./train2017
 
        annFile = '{}\\annotations\\{}.json'.format(dataDir,dataType)  # ./annotations/instances_val2017.json  ./annotations/instances_train2017.json
        coco = COCO(annFile)
        n_imgs = len(coco.imgs)  # 5000
        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:  # 实现多进程通信
            fs = [executor.submit(crop_img, coco.loadImgs(id)[0],
                                  coco.loadAnns(coco.getAnnIds(imgIds=id, iscrowd=None)),
                                  set_crop_base_path, set_img_base_path, instanc_size) for id in coco.imgs]
            for i, f in enumerate(futures.as_completed(fs)):
                # Write progress to error so that it can be seen
                printProgress(i, n_imgs, prefix=dataType, suffix='Done ', barLength=40)
    print('done')
 
if __name__ == '__main__':
    since = time.time()
    # main(int(sys.argv[1]), int(sys.argv[2]))
    main()
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))