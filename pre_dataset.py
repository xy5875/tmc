import os
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from tqdm import tqdm


# class CustomDataset(Dataset):
#     def __init__(self, root_folder, transform=None):
#         self.root_folder = root_folder
#         self.transform = transform
#         self.labels = [int(label) for label in os.listdir(root_folder)]
#         self.image_paths = []

#         for label in self.labels:
#             label_folder = os.path.join(root_folder, str(label))
#             image_list = os.listdir(label_folder)
#             image_paths = [os.path.join(label_folder, img) for img in image_list]
#             self.image_paths.extend(image_paths)
        
    

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, index):
#         image_path = self.image_paths[index]
#         label = int(image_path.split(os.path.sep)[-2])
#         image = Image.open(image_path).convert('RGB')

#         if self.transform:
#             image = self.transform(image)

#         return image, label


# 设置数据集根目录和转换


# class CustomDataset(Dataset):
#     _shared_datasets = {}  # 用于保存已加载的数据集

#     def __init__(self, dataset_name, id=-1):
#         self.dataset_name = dataset_name
#         self.id = id 
#         self.transform = transforms.Compose([
#         transforms.ToTensor()
#         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#     ])
#         if dataset_name not in CustomDataset._shared_datasets:
#             CustomDataset._shared_datasets[dataset_name] = self.load_data()

#         self.data = CustomDataset._shared_datasets[dataset_name]

#     def load_data(self):
#         # 这里根据 dataset_name 加载对应的数据集
#         # 你可以根据需要进行修改
#         data = []  # 用于存储 (image, label) 对的列表
        
#         if self.dataset_name == "test":
#             root_folder = "/home/dell/xy/new/mobicom-TWT/cifar/test_data/0"  # 修改为你的数据集路径

#             labels = [int(label) for label in os.listdir(root_folder)]

#             for label in labels:
#                 label_folder = os.path.join(root_folder, str(label))
#                 image_list = os.listdir(label_folder)
                
#                 for img_name in image_list:
#                     image_path = os.path.join(label_folder, img_name)
#                     image = Image.open(image_path).convert('RGB')

#                     if self.transform:
#                         image = self.transform(image)

#                     label = torch.tensor(int(label))  # 转换为 PyTorch 张量
#                     image = image.cuda(cuda)  # 将图像移到 GPU 上
#                     label = label.cuda(cuda)  # 将标签移到 GPU 上

#                     data.append((image, label))
#         elif self.dataset_name == "eztrain":
#             root_folder = "/home/dell/xy/new/mobicom-TWT/cifar/ez_data/0"  # 修改为你的数据集路径

#             labels = [int(label) for label in os.listdir(root_folder)]

#             for label in labels:
#                 label_folder = os.path.join(root_folder, str(label))
#                 image_list = os.listdir(label_folder)
                
#                 for img_name in image_list:
#                     image_path = os.path.join(label_folder, img_name)
#                     image = Image.open(image_path).convert('RGB')

#                     if self.transform:
#                         image = self.transform(image)

#                     label = torch.tensor(int(label))  # 转换为 PyTorch 张量
#                     image = image.cuda(cuda)  # 将图像移到 GPU 上
#                     label = label.cuda(cuda)  # 将标签移到 GPU 上

#                     data.append((image, label))
        
#         else :
#             root_folder = f"/home/dell/xy/new/mobicom-TWT/cifar/train_data_0.1/{self.id}"  # 修改为你的数据集路径

#             labels = [int(label) for label in os.listdir(root_folder)]

#             for label in labels:
#                 label_folder = os.path.join(root_folder, str(label))
#                 image_list = os.listdir(label_folder)
                
#                 for img_name in image_list:
#                     image_path = os.path.join(label_folder, img_name)
#                     image = Image.open(image_path).convert('RGB')

#                     if self.transform:
#                         image = self.transform(image)

#                     label = torch.tensor(int(label))  # 转换为 PyTorch 张量
#                     image = image.cuda(cuda)  # 将图像移到 GPU 上
#                     label = label.cuda(cuda)  # 将标签移到 GPU 上

#                     data.append((image, label))
#         return data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         return self.data[index]


class CustomDataset(Dataset):
    _shared_datasets = {}  # 用于保存已加载的数据集

    def __init__(self, dataset_name, id=-1,cuda=1,test_flie = ''):
        self.dataset_name = dataset_name
        self.id = id 
        self.testflie = test_flie
        self.transform = transforms.Compose([
        transforms.ToTensor()
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
        if dataset_name == "test":
            #  CustomDataset._shared_datasets[dataset_name] = self.load_data(cuda)
            self.data = self.load_data(cuda)
        else:
            if dataset_name not in CustomDataset._shared_datasets:
                CustomDataset._shared_datasets[dataset_name] = self.load_data(cuda)
                self.data = CustomDataset._shared_datasets[dataset_name]
            else :
                self.data = CustomDataset._shared_datasets[dataset_name]

    def load_data(self,cuda):
        # 这里根据 dataset_name 加载对应的数据集
        # 你可以根据需要进行修改
        data = []  # 用于存储 (image, label) 对的列表
        
        
        if self.dataset_name == "test":
            root_folder = self.testflie  # 修改为你的数据集路径

            labels = [int(label) for label in os.listdir(root_folder)]
            print(labels)
            for label in labels:
                label_folder = os.path.join(root_folder, str(label))
                image_list = os.listdir(label_folder)
                
                for img_name in image_list:
                    image_path = os.path.join(label_folder, img_name)
                    image = Image.open(image_path).convert('RGB')

                    if self.transform:
                        image = self.transform(image)

                    label = torch.tensor(int(label))  # 转换为 PyTorch 张量
                    image = image.cuda(cuda)  # 将图像移到 GPU 上
                    label = label.cuda(cuda)  # 将标签移到 GPU 上

                    data.append((image, label))
        elif self.dataset_name == "eztrain":
            root_folder = "/home/dell/xy/new/mobicom-TWT/cifar/ez_data/0"  # 修改为你的数据集路径

            labels = [int(label) for label in os.listdir(root_folder)]

            for label in labels:
                label_folder = os.path.join(root_folder, str(label))
                image_list = os.listdir(label_folder)
                
                for img_name in image_list:
                    image_path = os.path.join(label_folder, img_name)
                    image = Image.open(image_path).convert('RGB')

                    if self.transform:
                        image = self.transform(image)

                    label = torch.tensor(int(label))  # 转换为 PyTorch 张量
                    image = image.cuda(cuda)  # 将图像移到 GPU 上
                    label = label.cuda(cuda)  # 将标签移到 GPU 上

                    data.append((image, label))
        
        else :
            root_folder = f"/home/dell/xy/AFLvsGFL/data/{self.id}"  # 修改为你的数据集路径
            labels = [int(label) for label in os.listdir(root_folder)]

            for label in labels:
                label_folder = os.path.join(root_folder, str(label))
                image_list = os.listdir(label_folder)
                
                for img_name in image_list:
                    image_path = os.path.join(label_folder, img_name)
                    image = Image.open(image_path).convert('RGB')

                    if self.transform:
                        image = self.transform(image)

                    label = torch.tensor(int(label))  # 转换为 PyTorch 张量
                    image = image.cuda(cuda)  # 将图像移到 GPU 上
                    label = label.cuda(cuda)  # 将标签移到 GPU 上

                    data.append((image, label))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]




class TrainDataset(Dataset):
    def __init__(self,dataset_path,cuda=1):
        self.root_folder = dataset_path
        self.transform = transforms.Compose([
        transforms.ToTensor()
    ])
        self.data = self.load_data(cuda)

    def load_data(self,cuda):
        data = []
        labels = [int(label) for label in os.listdir(self.root_folder)]
        for label in labels:
            label_folder = os.path.join(self.root_folder, str(label))
            image_list = os.listdir(label_folder)
            
            for img_name in image_list:
                image_path = os.path.join(label_folder, img_name)
                image = Image.open(image_path).convert('RGB')

                if self.transform:
                    image = self.transform(image)

                label = torch.tensor(int(label))  # 转换为 PyTorch 张量
                image = image.cuda(cuda)  # 将图像移到 GPU 上
                label = label.cuda(cuda)  # 将标签移到 GPU 上

                data.append((image, label))
        return data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
