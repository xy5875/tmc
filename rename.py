# import os

# # 源文件夹路径
# source_folder = '/data/xy/TMC/data/train_all/0/9'

# # 获取源文件夹中所有图片文件的列表
# image_files = [f.path for f in os.scandir(source_folder) if f.is_file() and f.name.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

# # 初始化文件计数器
# file_counter = 1800

# # 遍历每个图片文件，并重命名
# for image_file in image_files:
#     file_name, file_extension = os.path.splitext(image_file)
#     new_file_name = f'image_{file_counter}{file_extension}'
#     new_file_path = os.path.join(os.path.dirname(image_file), new_file_name)
#     os.rename(image_file, new_file_path)
#     file_counter += 1

# print("图片已成功重命名！")

import os
import re

# 源文件夹路径
source_folder = '/data/xy/TMC/data/train_all'

# 获取源文件夹中所有图片文件的列表
image_files = [f.path for f in os.scandir(source_folder) if f.is_file() and f.name.endswith(('.jpg', '.jpeg', '.png', '.gif'))]

# 遍历每个图片文件，并重命名
for image_file in image_files:
    # 提取文件名和扩展名
    file_name, file_extension = os.path.splitext(image_file)
    
    # 使用正则表达式匹配数字部分
    match = re.search(r'(\d+)', os.path.basename(file_name))
    if match:
        new_file_name = match.group(1) + file_extension
        new_file_path = os.path.join(os.path.dirname(image_file), new_file_name)
        os.rename(image_file, new_file_path)

print("图片已成功重命名！")
