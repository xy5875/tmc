import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import os
import time
import pickle
import requests
import numpy as np
import torchvision
from client_global_variable import Client_Status
import traceback
import logging
import argparse
from pre_dataset import TrainDataset
from model import ResNet34 as Model
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from tqdm import tqdm




#######################################ARGS##################################################

#######################################DATAloader##################################################
transform = transforms.Compose([
    transforms.ToTensor()  # 将 PIL 图像或 numpy.ndarray 转换为 tensor
      # 归一化
])

# 加载训练数据集
train_dataset = torchvision.datasets.CIFAR10(root='/home/dell/xy/AFLvsGFL/data/cifar', train=True,
                                               download=False, transform=transform)
# 创建训练数据集 DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 加载测试数据集
test_dataset = torchvision.datasets.CIFAR10(root='/home/dell/xy/AFLvsGFL/data/cifar', train=False,
                                              download=False, transform=transform)
# 创建测试数据集 DataLoader
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


def evaluate_model(model, dataloader):
    model.eval()  # 将模型设置为评估模式
    correct = 0
    total = 0
    with torch.no_grad():  # 禁用梯度计算
        for images, labels in dataloader:
            images = images.cuda(0)
            labels = labels.cuda(0)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# 在测试数据集上评估模型


#######################################TRAIN##################################################
net = Model()
net = net.cuda(0)


# model_path = args.model_path

# # 加载模型参数

# net.load_state_dict(torch.load(model_path))


# for param in net.parameters():
#     param.requires_grad = False

# # Unfreeze output layer parameters
# for param in net.fc.parameters():
#     param.requires_grad = True


cross_entropy = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr = 0.01)
max_acc = 0

# acc = evaluate_model(net, loader)
# print("init acc is ",acc)
for epoch in range(600): 
    net = net.train()           
    for batch_num,(image,label) in enumerate(train_loader):
       
        image = image.cuda(0)
        label = label.cuda(0)
        output = net(image)
        entropy_num = cross_entropy(output,label)
        loss = entropy_num
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
    test_accuracy = evaluate_model(net, test_loader)
    if test_accuracy > max_acc:
        max_acc = test_accuracy
        torch.save(net.state_dict(), "/home/dell/xy/AFLvsGFL/model/pt/1.pth")
    print("epoch is ",epoch,"  ","max acc is ",max_acc,"  ","now acc is ",test_accuracy)






