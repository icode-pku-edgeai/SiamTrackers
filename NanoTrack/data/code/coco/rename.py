
"""
# txt改名
# """
# import os
# from PIL import Image
# import glob


# image_input_dir = 'D:\\datasets\\new_dataset'
# txt_input_dir = 'D:\\datasets\\new_dataset_labels'
# # 修改为你想要存储jpg图片的路径
# image_output_dir = 'D:\\datasets\\new_dataset_rename'
# # 修改为你想要存储txt的路径
# txt_output_dir = 'D:\\datasets\\new_dataset_rename_labels'

# # 保证目标文件夹存在
# if not os.path.exists(image_output_dir):
#     os.makedirs(image_output_dir)
# if not os.path.exists(txt_output_dir):
#     os.makedirs(txt_output_dir)

# # 获取png文件列表
# png_files = [f for f in os.listdir(image_input_dir) if f.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.txt'))]

# # 按照文件名排序
# sorted_png_files = sorted(png_files)

# # 批量转换并重命名
# for i, png_file in enumerate(sorted_png_files):
#     img = Image.open(os.path.join(png_dir, png_file))
#     rgb_img = img.convert('RGB')
#     jpg_file = os.path.join(jpg_dir, f'{i+45453}.jpg')
#     rgb_img.save(jpg_file, 'JPEG')
#     # with open(os.path.join(png_dir, png_file), "r") as f_in:
#     #     lines = f_in.readlines()
    # txt_file = os.path.join(jpg_dir, f'{i+45453}.txt')
    # with open(txt_file, "w") as f_out:
    #     f_out.writelines(lines)

# import os  
# from PIL import Image  
# import shutil  
  
# def rename_and_save_images(source_folder, target_folder, new_name_prefix=""):  
#     """  
#     批量加载图片，重命名并保存到指定文件夹。  
  
#     :param source_folder: 源文件夹路径，包含要处理的图片。  
#     :param target_folder: 目标文件夹路径，用于保存重命名后的图片。  
#     :param new_name_prefix: 新文件名的前缀，默认为"image_"。  
#     """  
#     # 确保目标文件夹存在  
#     if not os.path.exists(target_folder):  
#         os.makedirs(target_folder)  
  
#     # 遍历源文件夹中的所有文件  
#     for filename in os.listdir(source_folder):  
#         # 跳过非图片文件（这里以.jpg和.png为例，您可以根据需要添加更多扩展名）  
#         if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')):  
#             continue  
  
#         # 完整路径  
#         source_path = os.path.join(source_folder, filename)  
#         # 构造新文件名，这里简单使用序号+前缀+原扩展名  
#         # 注意：这里为了简单起见，序号从1开始，并递增，没有考虑可能的并发冲突  
#         counter = 45453  
#         base_name, ext = os.path.splitext(filename)  
#         new_filename = f"{new_name_prefix}{counter:03d}{ext}"  
#         # 检查新文件名是否已存在，如果存在则递增计数器  
#         while os.path.exists(os.path.join(target_folder, new_filename)):  
#             counter += 1  
#             new_filename = f"{new_name_prefix}{counter:03d}{ext}"  
  
#         # 目标路径  
#         target_path = os.path.join(target_folder, new_filename)  
  
#         # 使用Pillow加载图片并保存，以保持原始质量  
#         with Image.open(source_path) as img:  
#             img.save(target_path)  
  
#         print(f"已保存: {target_path}")  
  
# # 使用示例  
# source_folder = 'D:\\datasets\\new_dataset_labels'  
# target_folder = 'D:\\datasets\\new_dataset_rename_labels'  
# rename_and_save_images(source_folder, target_folder)



import os  
from pathlib import Path  
  
def rename_and_replace_jpg_files(image_path, label_path, start_number):  
    # 确保提供的文件夹路径是存在的  
    if not os.path.isdir(image_path):  
        print(f"Error: The directory {image_path} does not exist.")  
        return 
    if not os.path.isdir(label_path):  
        print(f"Error: The directory {label_path} does not exist.")  
        return  
  
    # 遍历文件夹下的所有文件  
    for filename in os.listdir(image_path):  
        # 检查文件是否是jpg文件  
        if filename.lower().endswith(('.jpg','.JPG', '.jpeg', '.png', '.gif', '.bmp', '.tiff')):  
            # 构建文件完整路径  
            old_file_path = os.path.join(image_path, filename)
            # 生成新文件名  
            new_filename = f"{start_number:06d}.jpg"  # 假设我们想要至少4位数的文件名，不足部分用0填充  
            new_file_path = os.path.join(image_path, new_filename)  
            # 增加数字以确保文件名唯一  
            while os.path.exists(new_file_path):  
                start_number += 1  
                new_filename = f"{start_number:06d}.jpg"  
                new_file_path = os.path.join(image_path, new_filename)  
  
            # 重命名文件（实际上是复制并删除原文件）  
            try:  
                Path(old_file_path).rename(new_file_path)  
                print(f"Renamed '{filename}' to '{new_filename}'")  
            except OSError as e:  
                print(f"Error: {e.strerror} when renaming {old_file_path} to {new_file_path}")  
        if filename.lower().endswith(('.jpg','.JPG', '.jpeg', '.png', '.gif', '.bmp', '.tiff')):  
                # 构建新的文件名  
            txt_name = filename.rsplit('.', 1)[0] + '.txt' 
        
        if txt_name.lower().endswith('.txt'):  
            # 构建文件完整路径  
            old_file_path = os.path.join(label_path, txt_name)
            # 生成新文件名  
            new_filename = f"{start_number:06d}.txt"  # 假设我们想要至少4位数的文件名，不足部分用0填充  
            new_file_path = os.path.join(label_path, new_filename)  
            # 增加数字以确保文件名唯一  
            while os.path.exists(new_file_path):  
                start_number += 1  
                new_filename = f"{start_number:06d}.txt"  
                new_file_path = os.path.join(label_path, new_filename)  
  
            # 重命名文件（实际上是复制并删除原文件）  
            try:  
                Path(old_file_path).rename(new_file_path)  
                print(f"Renamed '{txt_name}' to '{new_filename}'")  
            except OSError as e:  
                print(f"Error: {e.strerror} when renaming {old_file_path} to {new_file_path}") 
  
# 使用示例  
image_path = "C:\\Users\\li\\Desktop\\repo\\SiamTrackers-master\\NanoTrack\\tank\\val\\1"  # 替换为你的文件夹路径  
label_path = "C:\\Users\\li\\Desktop\\repo\\SiamTrackers-master\\NanoTrack\\tank\\labels\\val_labels"  # 替换为你的文件夹路径  
start_number_to_use = 0  # 从哪个数字开始编排文件名  
rename_and_replace_jpg_files(image_path, label_path,start_number_to_use)