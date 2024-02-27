import os
import random
import shutil

def copy_random_images(source_folder, destination_folder, n):
    # 检查目标文件夹是否存在，如果不存在则创建它
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 获取源文件夹中的所有图片文件
    image_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

    # 如果源文件夹中没有图片文件，则退出函数
    if not image_files:
        print("源文件夹中没有图片文件。")
        return

    # 如果n大于源文件夹中的图片数量，则将n设置为源文件夹中的图片数量
    n = min(n, len(image_files))

    # 从源文件夹中随机选择n张图片
    selected_images = random.sample(image_files, n)

    # 复制选定的图片到目标文件夹中
    for image_file in selected_images:
        source_path = os.path.join(source_folder, image_file)
        destination_path = os.path.join(destination_folder, image_file)
        shutil.copyfile(source_path, destination_path)
        print(f"复制文件: {image_file} -> {destination_folder}")

if __name__ == "__main__":
    client_labels = {
    0: [0, 8],
    1: [0, 8],
    2: [1, 3],
    3: [1, 3],
    4: [2, 6],
    5: [2, 6],
    6: [4, 5],
    7: [4, 5],
    8: [7, 9],
    9: [7, 9]
}

    
    
    # 源文件夹路径和目标文件夹路径

    source_folder = f"/home/dell/xy/AFLvsGFL/data/9/9"
    destination_folder = f"/home/dell/xy/AFLvsGFL/data/drift_ez/E/9"

    # 需要复制的图片数量
    n = 500

    # 调用函数复制图片
    copy_random_images(source_folder, destination_folder, n)
