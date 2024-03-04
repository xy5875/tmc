import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet34
from torchvision.datasets import ImageFolder
from tqdm import tqdm


class CustomDataset(Dataset):
    def __init__(self, root_folder, transform=None):
        self.root_folder = root_folder
        self.transform = transform
        self.labels = [int(label) for label in os.listdir(root_folder)]
        self.image_paths = []

        for label in self.labels:
            label_folder = os.path.join(root_folder, str(label))
            image_list = os.listdir(label_folder)
            image_paths = [os.path.join(label_folder, img) for img in image_list]
            self.image_paths.extend(image_paths)
        
    

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = int(image_path.split(os.path.sep)[-2])
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


# 设置数据集根目录和转换


# class CustomDataset(Dataset):
#     def __init__(self, root_folder, transform=None):
#         self.root_folder = root_folder
#         self.transform = transform
#         self.data = []  # 用于存储 (image, label) 对的列表

#         labels = [int(label) for label in os.listdir(root_folder)]

#         for label in labels:
#             label_folder = os.path.join(root_folder, str(label))
#             image_list = os.listdir(label_folder)
            
#             for img_name in image_list:
#                 image_path = os.path.join(label_folder, img_name)
#                 image = Image.open(image_path).convert('RGB')

#                 if self.transform:
#                     image = self.transform(image)

#                 label = torch.tensor(int(label))  # 转换为 PyTorch 张量
#                 image = image.cuda(1)  # 将图像移到 GPU 上
#                 label = label.cuda(1)  # 将标签移到 GPU 上

#                 self.data.append((image, label))

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         return self.data[index]




