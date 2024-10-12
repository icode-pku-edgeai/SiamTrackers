from coco import COCO
from os.path import join
import json
 
dataDir = 'C:\\Users\\li\\Desktop\\repo\\SiamTrackers-master\\NanoTrack\\data\\tank'
count = 0
for dataType in ['val', 'train']:
    dataset = dict()
    annFile = '{}\\annotations\\{}.json'.format(dataDir, dataType)  # './annotations/instances_val2017.json'
    coco = COCO(annFile)  # <c>
    n_imgs = len(coco.imgs)  # 5000
    for n, img_id in enumerate(coco.imgs):  # img_id 举例 397133 拿出每个图片的所有的标注信息
        # print('subset: {} image id: {:04d} / {:04d}'.format(dataType, n, n_imgs))
        img = coco.loadImgs(img_id)[0]  # <c>
        annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)  # <c>
        anns = coco.loadAnns(annIds)  # <c>
        video_crop_base_path = join(dataType, img['file_name'].split('/')[-1].split('.')[0])  # 举例 'val2017/000000397133'
 
        if len(anns) > 0:
            dataset[video_crop_base_path] = dict()  #  {'val2017/000000397133':{}}
 
        for trackid, ann in enumerate(anns):  # ann <c> 相当于拿出每张图片的每个目标的bbox
            rect = ann['bbox']  # bbox
            c = ann['category_id']  # 类别 id
            bbox = [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]  # xywh to xyxy
            if rect[2] <= 0 or rect[3] <= 0:  # lead nan error in cls.
                count += 1
                print(count, rect)
                continue
            dataset[video_crop_base_path]['{:02d}'.format(trackid)] = {'000000': bbox}  # 举例 {'val2017/000000397133':{'00':{'000000':[217.62,240.54,256.61,298.289999]}}}
 
    print('save json (dataset), please wait 20 seconds~')
    json.dump(dataset, open('{}.json'.format(dataType), 'w'), indent=4, sort_keys=True)  # 将字典写入json文件，indent表示前边间隔长度，sort_key 表示按键顺序进行排列
    print('done!')