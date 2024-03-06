import os
from PIL import Image
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

# 创建存储图像的文件夹
output_folder_root = '/data/xy/TMC/data/TRAIN'
if not os.path.exists(output_folder_root):
    os.makedirs(output_folder_root)

transform = transforms.Compose([
    transforms.ToTensor(),
])

# 加载CIFAR-10数据集
cifar10_train_dataset = datasets.CIFAR10(root='/data/xy/TMC/data/all', train=True, download=False, transform=transform)

# 存储每个类别的图像数
images_per_class = [0] * 10
j = 0
for m in range(10):
    # 将训练集图像转换为图片并保存
    for i, (image, label) in enumerate(cifar10_train_dataset):
        # 图像保存路径
        # class_folder = os.path.join(output_folder_root, str(label))
        # if not os.path.exists(class_folder):
        #     os.makedirs(class_folder)

        if label == m:

            # 图像保存路径
            image_path = os.path.join(output_folder_root, f'{j}.png')

            # 保存图像
            save_image(image, image_path)

            print(f'Saved image {i}.png')

            # 更新该类别已保存图像的数量
            j = j + 1

print('All images saved successfully.')
