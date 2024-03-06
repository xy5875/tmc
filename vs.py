import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import os
import time
import pickle
import requests
import numpy as np
from client_global_variable import Client_Status
import traceback
import logging
import argparse
from pre_dataset import TrainDataset
from model import ResNet34 as Model




#######################################ARGS##################################################
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_path',default='default',help='log file name')
parser.add_argument('--test_path',default='default',help='info path')
parser.add_argument('--model_path',default='default',help='model path')
args = parser.parse_args()

#######################################DATAloader##################################################
from pre_dataset import *  
dataset_path = args.train_path

custom_dataset = TrainDataset(dataset_path=dataset_path,cuda=0)
train_data_loader = DataLoader(dataset=custom_dataset, batch_size=10, shuffle=True,drop_last=True)

dataset_name = "test"
test_flie = args.test_path
custom_dataset = CustomDataset(dataset_name=dataset_name,cuda=0,test_flie = test_flie)
loader = DataLoader(dataset=custom_dataset, batch_size=32, shuffle=True)


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


model_path = args.model_path

# 加载模型参数

net.load_state_dict(torch.load(model_path))


for param in net.parameters():
    param.requires_grad = False

# Unfreeze output layer parameters
for param in net.fc.parameters():
    param.requires_grad = True


cross_entropy = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr = 0.01)
max_acc = 0

acc = evaluate_model(net, loader)
print("init acc is ",acc)
for epoch in range(300): 
    net = net.train()           
    for batch_num,(image,label) in enumerate(train_data_loader):
       
        image = image.cuda(0)
        label = label.cuda(0)
        output = net(image)
        entropy_num = cross_entropy(output,label)
        loss = entropy_num
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
    test_accuracy = evaluate_model(net, loader)
    if test_accuracy > max_acc:
        max_acc = test_accuracy
    print("epoch is ",epoch,"  ","max acc is ",max_acc,"  ","now acc is ",test_accuracy)






