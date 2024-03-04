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


parser = argparse.ArgumentParser(description='Client running')
parser.add_argument('--name','-n',help='client name')
parser.add_argument('--serverport','-p',help='server port',default='8080')
parser.add_argument('--serverip','-i',help='server ip',default='127.0.0.1')
parser.add_argument('--dataroot','-r',help='data root')
parser.add_argument('--datainfo',help='data information',default='data/info.json')
parser.add_argument('--cuda',help='cuda index',default='0')
parser.add_argument('--anchor',help='anchor dataset use')
parser.add_argument('--mask',help='use mask or not')
parser.add_argument('--seed')
parser.add_argument('--delay')
parser.add_argument('--testfile',help = "test file")
parser.add_argument('--taile')
import copy
def train_core(net:nn.Module,trainloader,cuda,delay,cfg,class_shot,anchor_loader=None,mask=None,taile=None)->nn.Module:
    net = net.cuda(cuda)
    net = net.train()
    
    if mask is not None:
        apply_freeze_mask_core(net,mask)
    
    
    cross_entropy = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr = cfg["lr"])
    
    total_loss_change = 0.0
    
    if taile == "False":
        for epoch in range(cfg["epoch"]):            
            for batch_num,(image,label) in enumerate(trainloader):
                origin_param = copy.deepcopy(net.state_dict())
                image = image.cuda(cuda)
                label = label.cuda(cuda)
                output = net(image)
                entropy_num = cross_entropy(output,label)
                loss = entropy_num
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if anchor_loader is not None:
                    for image,label in anchor_loader:
                        image = image.cuda(cuda)
                        label = label.cuda(cuda)
                        output = net(image)
                        entropy_num = cross_entropy(output,label)
                        loss = entropy_num
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
            time.sleep(delay)
    else:
        for epoch in range(cfg["epoch"]):            
            for batch_num,(image,label) in enumerate(trainloader):
                origin_param = copy.deepcopy(net.state_dict())
                image = image.cuda(cuda)
                label = label.cuda(cuda)
                optimizer.zero_grad()
                output = net(image)
                
                loss = cross_entropy(output, label)
                loss.backward()
                with torch.no_grad():
                    grad_norm = torch.norm(torch.cat([p.grad.view(-1) for p in net.parameters()]))
                    #loss_change = taylor_approximation(grad_norm, optimizer.param_groups[0]['lr'])
                    loss_change = -1 * optimizer.param_groups[0]['lr'] * torch.norm(grad_norm)**2
                    total_loss_change += loss_change.item()
                optimizer.step()
            print("delay is :",delay)   
            time.sleep(delay)            
            
    return net,total_loss_change,class_shot

def mask_build_core(model,seed ,freeze_prob=0.1):
    np.random.seed(seed)
    freeze_mask = {}
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            num_channels = layer.weight.size(0)
            # 为每个通道生成一个随机数，然后根据freeze_prob决定是否冻结该通道
            channel_mask = np.random.rand(num_channels) < freeze_prob
            freeze_mask[name] = channel_mask
    return freeze_mask

def apply_freeze_mask_core(model, freeze_mask):
    with torch.no_grad():
        for name, layer in model.named_modules():
            if name in freeze_mask and isinstance(layer, torch.nn.Conv2d):
                channel_mask = freeze_mask[name]
                for i, freeze in enumerate(channel_mask):
                    if freeze:
                        # 冻结该通道的权重和偏置
                        layer.weight[i].requires_grad = False
                        # layer.weight[i].fill_(0)
                        # print(layer.weight[i])
                        if layer.bias is not None:
                            layer.bias[i].requires_grad = False
                            layer.bias[i].fill_(0)
                    else:
                        layer.weight[i].requires_grad = True

def gene_mask_index_core(tensor):
    unique_elements, counts = torch.unique(tensor, return_counts=True)
    max_count_index = torch.argmax(counts)
    most_frequent_element = unique_elements[max_count_index]
    return most_frequent_element.item()

def apply_mask_core(mask,origin_param,new_param):
    for name in mask.keys():
        origin_param[name] = origin_param[name]*(1-mask[name])+new_param[name]*mask[name]
        
def compute_accuracy(possibility, label):
    sample_num = label.size(0)
    _, index = torch.max(possibility, 1)
    correct_num = torch.sum(label == index)
    return (correct_num/sample_num).item()

def test(net,loader,name):
    net.eval()
    acc = 0
    num = 0
    for batch_num,(image,label) in enumerate(loader):
        num += 1
        image = image.cuda(Client_Status.CUDA)
        label = label.cuda(Client_Status.CUDA)
        output = net(image)
        acc+= compute_accuracy(output,label)
    acc/= num
    if not(name in Client_Status.MAX_ACC.keys()):
        Client_Status.MAX_ACC[name] = acc
    elif acc>Client_Status.MAX_ACC[name]:
        Client_Status.MAX_ACC[name] = acc
    logging.info("{:s} max accuracy: {:.2f} current accuracy: {:.2f}"
              .format(name, Client_Status.MAX_ACC[name],acc*100))


from pre_dataset import *   
def test_loader_build_core(cuda,path):
    dataset_name = "test"
    custom_dataset = CustomDataset(dataset_name=dataset_name,cuda=cuda,test_flie = path)
    loader = DataLoader(dataset=custom_dataset, batch_size=32, shuffle=True)
    return loader

def ezloader_build_core(cuda):
    dataset_name = "eztrain"
    custom_dataset = CustomDataset(dataset_name=dataset_name,cuda=cuda)
    loader = DataLoader(dataset=custom_dataset, batch_size=10, shuffle=True,drop_last=False)
    return loader

def url_build_core(serverIp,serverPort):
    return f'http://{serverIp}:{serverPort}/server'

def req_model_core(serverIp,serverPort):
    url = f'{url_build_core(serverIp,serverPort)}/req_model'
    r = requests.post(url,data=pickle.dumps(''))
    return pickle.loads(r.content)



def req_cfg_core(serverIp,serverPort):
    url = f'{url_build_core(serverIp,serverPort)}/req_cfg'
    r = requests.get(url)
    
    return pickle.loads(r.content)

# def info_build_core(datainfo):
#     with open(datainfo) as f:
#         info = json.loads(f.read())
#     return info

def send_model_core(serverIp,serverPort,model,name,cuda,fisher,time,class_shot):
    # print("class shot is ",class_shot)
    url = f'{url_build_core(serverIp,serverPort)}/send_model'
    data = {
        'name':name,
        'model':model,
        'cuda':cuda,
        'fisher':fisher,
        'time':time,
        'class_shot':class_shot
    }
    data = pickle.dumps(data)
    resp = requests.post(url,data)
    return resp.content

def masks_build_core(serverIp,serverPort,cuda):
    url = f"{url_build_core(serverIp,serverPort)}/req_mask"
    r = requests.get(url)
    masks = pickle.loads(r.content)
    for mask in masks:
        for key in mask.keys():
            mask[key] = mask[key].cuda(cuda)
    return masks

def req_class_list_core(serverIp,serverPort):
    url = f'{url_build_core(serverIp,serverPort)}/req_class'
    r = requests.post(url,data=pickle.dumps(''))
    
    return pickle.loads(r.content)


from pre_dataset import *  
import time
def train_loader_build_core(cuda,dataset_path,batch_size,list):

    custom_dataset = TrainDataset(dataset_path=dataset_path,list=list)
    class_shot = custom_dataset.class_shot
    train_data_loader = DataLoader(dataset=custom_dataset, batch_size=batch_size, shuffle=True,drop_last=False)
  

    return train_data_loader,class_shot

def trans_bool_value(value):
    if value=='True':
        return True
    else:
        return False

class Client():
    def __init__(self, args) -> None:
        self.init_pass_(args)
    
    def init_pass_(self, args):
        
        self.name = args.name
        self.port = args.serverport
        self.ip = args.serverip
        self.dataroot = args.dataroot
        self.datainfo = args.datainfo
        self.cuda = int(args.cuda)
        self.anchor = trans_bool_value(args.anchor)
        self.use_mask = trans_bool_value(args.mask)
        self.seed = int(args.seed)
        self.delay = int(args.delay)
        self.testfile = args.testfile
        self.fisher = 0
        self.cfg = self.req_cfg()
        self.model = self.req_model()
        
        self.train_class_list = self.req_class_list()
        
        self.train_loader ,self.class_shot = self.train_loader_build()
        #self.test_loader = self.test_loader_build()
        self.anchor_loader = self.anchor_loader_build()
        self.mask = self.mask_build()
        self.taile = args.taile
        self.time = 0
       
        
        
    def run_train(self):
        
        t1=time.time()
        print("client id is : ",self.name)
        model = self.train()
        model.cpu()
        self.send_model(model)
        t2=time.time()
        #self.time = t2-t1
        print("time is ",t2-t1)
    
    def req_model(self):
        return req_model_core(self.ip,self.port)
    
    def req_class_list(self):
        return req_class_list_core(self.ip,self.port)
    
    def req_cfg(self):
        return req_cfg_core(self.ip,self.port)
    
    def train_loader_build(self):
        dataset_path = f'{self.dataroot}/{self.name}'
        print("dataset_path is ",dataset_path)
        train_loader,class_shot = train_loader_build_core(self.cuda,dataset_path,self.cfg['batch_size'],self.train_class_list)
        return train_loader,class_shot
    
    def test_loader_build(self):
        return test_loader_build_core(self.cuda,self.testfile)
    
    def anchor_loader_build(self):
        print(self.anchor)
        if self.anchor:
            return ezloader_build_core(self.cuda)
        else:
            return None
    
    
    def train(self):
        t1=time.time()
        print("train in class shot is ",self.class_shot)
        net,self.fisher,self.class_shot = train_core(self.model,self.train_loader,self.cuda,self.delay,self.cfg,self.class_shot,self.anchor_loader,self.mask,self.taile)
        t2=time.time()
        self.time = t2-t1
        return net
    
    def send_model(self,model):
        return send_model_core(self.ip,self.port,model,self.name,self.cuda,self.fisher,self.time,self.class_shot)
    
    def mask_build(self):
        if self.use_mask:
            return mask_build_core(self.model,self.seed,0.9)
        else:
            return None
    
        
    

if __name__ == "__main__":
    args = parser.parse_args()
    client = Client(args)
    client.run_train()

    