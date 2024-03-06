import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
from model import ResNet34 as Model
import torch.nn as nn

class CustomDatasetA(Dataset):
    def __init__(self, image_folder, label_file, transform=None):
        self.image_folder = image_folder
        self.label_file = label_file
        self.transform = transform

        # 读取标签文件
        self.labels = self.read_labels()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, f'{idx}.png')
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, label

    def read_labels(self):
        labels = []
        with open(self.label_file, 'r') as f:
            for line in f:
                label_str = line.strip().split('acc: ')[1]
                label = np.array(eval(label_str))  # 将字符串转换为列表
                labels.append(label)
        return labels

def create_dataloader(image_folder, label_file, batch_size, num_workers):
    # 定义数据预处理
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # 创建数据集
    dataset = CustomDatasetA(image_folder, label_file, transform=transform)

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader

# 使用示例
image_folder = '/data/xy/TMC/data/TRAIN'
label_file = '/data/xy/TMC/data/LABEL.txt'
batch_size = 64
num_workers = 4

train_dataloader = create_dataloader(image_folder, label_file, batch_size, num_workers)
# 迭代数据加载器并输出标签

from pre_dataset import CustomDataset

dataset_name = "test"
path = os.path.join('/data/xy/TMC/data/test_avg/0')
#print("test path is ",path)
custom_dataset = CustomDataset(dataset_name=dataset_name,cuda=3,test_flie = path)
test_loader = DataLoader(dataset=custom_dataset, batch_size=128, shuffle=True)





import torch.optim as optim



model = Model().cuda(3)
# 创建模型、损失函数和优化器

criterion = nn.L1Loss()  # 使用 L1 损失作为 MAE 损失
optimizer = optim.Adam(model.parameters(), lr=0.0001)

def test_model(model, test_loader):
    model.eval()  # 将模型设置为评估模式
    correct = 0
    total = 0

    with torch.no_grad():  # 不进行梯度计算
        for images, labels in test_loader:
            images, labels = images.to(3), labels.to(3)

            outputs = model(images)

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 输出测试结果
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
    return accuracy
    # 打开结果文件
    

import torch.nn.functional as F

# 计算平均绝对误差损失函数
def calculate_mae_loss(outputs, labels):
    batch_size = outputs.size(0)
    # print("---------------------------")
    # print("outputs",outputs)
    # print("---------------------------")
    # print("labels",labels)
    # print("---------------------------")
    # print("outputs-la",outputs - labels)
    # print("---------------------------")
    mae_loss = torch.abs(outputs - labels).sum() / batch_size
    # print("mae = ",mae_loss)
    # print("---------------------------")
    return mae_loss

# 训练模型
num_epochs = 1000
ACC=[]
for epoch in range(num_epochs):
    print("epoch is ",epoch)
    model.train()
    running_loss = 0.0
    for images, labels in train_dataloader:
        images= images.cuda(3)
        labels=labels.cuda(3)
        optimizer.zero_grad()
        outputs = model(images)
        outputs = F.softmax(outputs, dim=1)
        loss = calculate_mae_loss(outputs, labels)
        
        loss.backward()
        optimizer.step()

    acc = test_model(model, test_loader)
    ACC.append(acc)
    with open(r'/data/xy/TMC/method_1/zhengliu_all.txt', 'w') as f:
        f.write(f'acc = {ACC}\n')




