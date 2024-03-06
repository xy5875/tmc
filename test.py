import threading
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from model2 import ResNet34 as Model
from flask import Blueprint, request
import json
import time
import pickle
from server_global_variable import Server_Status
import logging
import copy
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import torch.nn.functional as F

# 定义数据路径
data_folder = '/data/xy/TMC/data/test/1'

# 定义模型路径
model_folder = '/data/xy/TMC/model'

# 定义结果保存路径
result_file = '/data/xy/TMC/data/LABEL.txt'

# 定义预处理转换
transform = transforms.Compose([
    transforms.ToTensor()
])

# 加载模型到GPU
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")


from pre_dataset import CustomDataset
dataset_name = "test"
path = os.path.join(data_folder)
#print("test path is ",path)
custom_dataset = CustomDataset(dataset_name=dataset_name,cuda=0,test_flie = path)
loader = DataLoader(dataset=custom_dataset, batch_size=128, shuffle=True)




models = []
for i in range(10):
    model_path = os.path.join(model_folder, f'class_{i}.pth')
    model = Model()
    model.load_state_dict(torch.load(model_path))
    model.to(device)  # 将模型移动到GPU
    model.eval()
    models.append(model)


def compute_accuracy(output, label):
    predictions = output.argmax(dim=1)
    correct = (predictions == label).float()
    # True Positive (TP): Prediction is positive and label is also positive
    TP = ((predictions == 0) & (label == 0)).sum().item()
    # True Negative (TN): Prediction is negative and label is also negative
    TN = ((predictions == 1) & (label == 1)).sum().item()
    # False Positive (FP): Prediction is positive but label is negative
    FP = ((predictions == 0) & (label == 1)).sum().item()
    # False Negative (FN): Prediction is negative but label is positive
    FN = ((predictions == 1) & (label == 0)).sum().item()

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return accuracy


net = models[1]
net.eval()  
net = net.cuda(0)

acc = 0

for batch_num, (image, label) in enumerate(loader):
    image = image.cuda()
    label = label.cuda()
    output = net(image)
    acc += compute_accuracy(output, label)
acc /= (batch_num + 1)  # 计算精度
print(acc)





# # 打开结果文件
# with open(result_file, 'w') as f:
#     # 遍历数据并将其移动到GPU
#     for i in range(50000):
#         image_path = os.path.join(data_folder, f'{i}.png')
#         image = Image.open(image_path).convert('RGB')
#         image = transform(image).unsqueeze(0)
#         image = image.to(device)  # 将数据移动到GPU

#         # 预测并保存结果
#         predictions = []
#         for model in models:
#             with torch.no_grad():
#                 outputs = model(image)
#                 outputs = F.softmax(outputs, dim=1)
#                 outputs = outputs.tolist()
#                 predictions.append(outputs[0][0])

#         f.write(f'image: {i}, acc: {predictions}\n')

print("预测结果已保存到文件中！")
# for i in range(2000):
#         image_path = os.path.join(data_folder, f'{i}.png')
#         image = Image.open(image_path).convert('RGB')
#         image = transform(image).unsqueeze(0)
#         image = image.to(device)  # 将数据移动到GPU
        
#         predictions = []
#         for model in models:
#             print("------------------------------------")
#         # 预测并保存结果
#             with torch.no_grad():
#                 outputs = model(image)
#                 outputs = outputs.tolist()
#                 print(outputs[0][0])

#             print("------------------------------------")
#                 #predictions.append(outputs[0])
