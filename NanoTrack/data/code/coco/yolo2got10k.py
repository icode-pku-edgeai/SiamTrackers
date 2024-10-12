import os  
  
def convert_yolo_to_bbox(x_center, y_center, w, h, img_width, img_height):  
    """  
    将YOLO格式的xywh转换为左上角坐标加宽度高度的格式  
    """  
    xmin = int(x_center * img_width - (w * img_width / 2))  
    ymin = int(y_center * img_height - (h * img_height / 2))  
    xmax = xmin + int(w * img_width)  
    ymax = ymin + int(h * img_height)  
    return xmin, ymin, xmax - xmin, ymax - ymin  # 返回xmin, ymin, width, height  
  
def read_yolo_labels_from_folder(folder_path, img_size):  
    """  
    从指定文件夹读取所有YOLO格式的txt标签文件，并转换格式  
    """  
    img_width, img_height = img_size  
    all_labels = []  
    for filename in os.listdir(folder_path):  
        if filename.endswith('.txt'):  
            filepath = os.path.join(folder_path, filename)  
            with open(filepath, 'r') as file:  
                lines = file.readlines()  
                for line in lines:  
                    parts = line.strip().split()  
                    if len(parts) < 5:  
                        continue  # 忽略格式不正确的行  
                    class_id, x_center, y_center, w, h = map(float, parts[:5])  
                    xmin, ymin, width, height = convert_yolo_to_bbox(x_center, y_center, w, h, img_width, img_height)
                    # xmin, ymin, width, height = float(xmin), float(ymin), float(width), float(height)  
                    all_labels.append(f"{xmin:.4f},{ymin:.4f},{width:.4f},{height:.4f},")  
                # all_labels=all_labels[:-1]
                all_labels.append(f"\n") 
                
    return all_labels  
  
def write_groundtruth_file(labels, output_filepath):  
    """  
    将转换后的标签写入到groundtruth.txt文件中  
    """  
    with open(output_filepath, 'w') as file:  
        file.writelines(labels)  
  
# 使用示例  
folder_path = 'C:\\Users\\li\\Desktop\\repo\\SiamTrackers-master\\NanoTrack\\tank\\val_labels'  # YOLO标签文件所在的文件夹路径  
img_size = (640, 480)  # 假设所有图像都是这个尺寸  
output_filepath = 'groundtruth.txt'  # 输出文件的路径  
  
labels = read_yolo_labels_from_folder(folder_path, img_size)  
write_groundtruth_file(labels, output_filepath)  
  
print(f"Ground truth data has been written to {output_filepath}")

# import os  
  
# def parse_yolo_label(file_path, img_width, img_height):  
#     """  
#     解析YOLO格式的txt标签文件，并返回边界框的左上角坐标和宽度高度（像素）。  
#     """  
#     with open(file_path, 'r') as file:  
#         lines = file.readlines()  
#     boxes = []  
#     for line in lines:  
#         parts = line.strip().split()  
#         if len(parts) < 5:  
#             continue  # 忽略不符合格式的行  
#         class_id, x_center, y_center, w, h = map(float, parts[:5])  
#         # 转换到左上角坐标和宽度高度  
#         xmin = round(x_center * img_width - (w * img_width / 2), 4)  
#         ymin = round(y_center * img_height - (h * img_height / 2), 4)  
#         xmax = round(xmin + w * img_width, 4)  
#         ymax = round(ymin + h * img_height, 4)  
#         boxes.append(f"{xmin} {ymin} {xmax-xmin} {ymax-ymin}\n")  
#     return boxes  
  
# def write_groundtruth(output_file, boxes):  
#     """  
#     将边界框信息写入到指定的txt文件中。  
#     """  
#     with open(output_file, 'w') as file:  
#         file.writelines(boxes)  
  
# def main(input_dir, output_file, img_width, img_height):  
#     """  
#     主函数，批量处理文件夹中的YOLO格式txt文件。  
#     """  
#     boxes = []  
#     for filename in os.listdir(input_dir):  
#         if filename.endswith('.txt'):  
#             file_path = os.path.join(input_dir, filename)  
#             parsed_boxes = parse_yolo_label(file_path, img_width, img_height)  
#             boxes.extend(parsed_boxes)  
      
#     write_groundtruth(output_file, boxes)  
  
# # 示例用法  
# input_directory = 'C:\\Users\\li\\Desktop\\repo\\SiamTrackers-master\\NanoTrack\\tank\\homemade2_labels'  # YOLO标签文件所在的文件夹路径  
# output_file = 'groundtruth.txt'  # 输出文件的名称  
# image_width = 640  # 图像的宽度  
# image_height = 480  # 图像的高度  
  
# main(input_directory, output_file, image_width, image_height)
