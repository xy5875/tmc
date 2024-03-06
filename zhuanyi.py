import os
import shutil

# 源文件夹路径
source_folder = '/data/xy/TMC/data/train_all/0'

# 目标文件夹路径
destination_folder = '/data/xy/TMC/data/train_all'

# 获取源文件夹中所有子文件夹的列表
subfolders = [f.path for f in os.scandir(source_folder) if f.is_dir()]

# 遍历每个子文件夹
for folder in subfolders:
    # 获取子文件夹中所有图片文件的列表
    image_files = [f.path for f in os.scandir(folder) if f.is_file() and f.name.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    
    # 移动图片文件到目标文件夹
    for image_file in image_files:
        shutil.move(image_file, destination_folder)

print("图片已成功移动到目标文件夹！")
