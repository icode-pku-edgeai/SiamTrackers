import cv2
import os
import matplotlib as plt

# save_step=1
# num=0
# data_dir='D:\\datasets\\origin_video\\'
# for filename in os.listdir(data_dir): 
#         # print('ok1')
#         if filename.lower().endswith('.mp4'): 
#             # print(filename)
#             video_dir=data_dir+filename
#             print(video_dir)
#             video=cv2.VideoCapture(video_dir)
#             while True:
#                 # print('ok3')
#                 ret, frame = video.read()
#                 if not ret:
#                     break
#                 num+=1
#                 # print('do')
#                 if num%save_step==0:
#                     # print('ok')
#                     cv2.imwrite("D:\\datasets\\new_dataset\\"+str(num+0)+".jpg",frame)



save_step=1
num=0

video=cv2.VideoCapture('D:\\datasets\\origin_video\\origin_video2\\1384975649-1-16.mp4')
count=1
count_name=-1
while True:
    # print('ok3')
    ret, frame = video.read()
    if not ret:
        break
    num+=1
    if num%50000==0:
        count+=1
        count_name=-1
    count_name+=1
    # print('do')    
    if num%save_step==0:
        # print('ok')
        image_path="C:\\Users\\li\\Desktop\\repo\\track\\SiamTrackers-master\\NanoTrack\\data\\origin_dataset\\10homemade_video\\000010"
        new_filename = f"{count_name:08d}.jpg"  
        new_filepath = f"{count:06d}"  
        new_filepath = os.path.join(image_path, new_filepath)
        if not os.path.exists(new_filepath):  
            os.makedirs(new_filepath)       

        new_file_path = os.path.join(new_filepath,new_filename)  
        cv2.imwrite(new_file_path,frame)

                