import threading
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from model import ResNet34 as Model
from flask import Blueprint, request
import json
import time
import pickle
from server_global_variable import Server_Status
import logging
import copy

server_app = Blueprint("server", __name__, url_prefix="/server")

server_status = Server_Status()


@server_app.route("/req_cfg", methods=["GET"])
def req_cfg():
    config = {
        "epoch": 2,
        "lr": 1e-2,
        "batch_size": 64
    }
    return pickle.dumps(config)


@server_app.route("/send_model", methods=["POST"])
def send_model():
    client_info = pickle.loads(request.data)
    net = client_info["model"]
    name = client_info['name']
    cuda = client_info['cuda']   
    server_status._instance_lock.acquire()
    send_model_core(net,name,cuda,server_status)
    server_status._instance_lock.release()
    return pickle.dumps("")

@server_app.route("/req_model", methods=["POST"])
def req_model():
    return server_status.MODEL_ENCODE

@server_app.route("/req_mask", methods=["GET"])
def req_mask():
    return server_status.MASKS

@server_app.route('/req_train',methods=['POST'])
def req_train():
    
    name = pickle.loads(request.data)
    server_status._instance_lock.acquire()
    train_tag = req_train_core(name,server_status)
    server_status._instance_lock.release()
    
    return train_tag




import shutil

def req_train_core(name,server_status:Server_Status):
    train_tag = {
        'train':False,
        'cuda':-1
    }
    name = int(name)
    # print("name i s ",name)
    # print("server_status.ROUND_NAMES",server_status.ROUND_NAMES)
    # print("name size is ",type(name))
    # print("round size is ",type(server_status.ROUND_NAMES[0]))
    if name in server_status.ROUND_NAMES:
        server_status.ROUND_NAMES.remove(name)
        print("server_status.ROUND_NAMES is ",server_status.ROUND_NAMES)
        server_status.TRAINING_NAMES.append(name)
        print("server_status.TRAINING_NAMES is ",server_status.TRAINING_NAMES)
        cuda = server_status.CUDA_LIST.pop()
        train_tag['train'],train_tag['cuda'] = True,cuda
    return pickle.dumps(train_tag)

def init_client_names(server_status:Server_Status,args):
    server_status.DATA_NAMES = args.client
def init_info_core(info_path,server_status:Server_Status):
    # with open(info_path,"r") as f:
    #     info = json.loads(f.read())
    server_status.DATA_INFO =  server_status.DATA_INFO 

def init_log_core(log_name):
    log_path = f'log_files/{log_name}.txt'
    f = open(log_path,'w')
    f.close()

def init_model_save(server_status:Server_Status,args):
    server_status.MODEL_SAVE=args.flie
    print(server_status.MODEL_SAVE)

def init_model_core(server_status:Server_Status):
    model = Model()
    server_status.MODEL_ENCODE = pickle.dumps(model)

def init_names_core(server_status:Server_Status,args):
    #server_status.DATA_NAMES = list(server_status.DATA_INFO.keys())
    server_status.DATA_NAMES = args.client

def init_parallel_number_core(server_status:Server_Status,args):
    server_status.PARALLEL_NUM = args.parallelnum
    server_status.PARALLEL_NUM = int(server_status.PARALLEL_NUM)
    assert server_status.PARALLEL_NUM == len(server_status.CUDA_LIST), "CUDA数目不对"

def init_test_flie(server_status:Server_Status,args):
    server_status.TEST_DATA_PATH = args.testfile

    
def send_model_core(model,name,cuda,server_status:Server_Status):
    asyn_round_core(model,name,cuda,server_status)
    
    
import random    
def asyn_round_core(model,name,cuda,server_status:Server_Status):
    main_model = pickle.loads(server_status.MODEL_ENCODE)
    server_status.ROUND+=1
   
    server_status.MODEL_ENCODE = pickle.dumps(aggregate_core(main_model,model))
    name = int(name)
    server_status.TRAINING_NAMES.remove(name)
   
    server_status.CUDA_LIST.append(cuda)
    train_able_pool = [name for name in server_status.DATA_NAMES if name not in server_status.TRAINING_NAMES]
    sample_num = server_status.PARALLEL_NUM-len(server_status.TRAINING_NAMES)-len(server_status.ROUND_NAMES)
    
    server_status.ROUND_NAMES = server_status.ROUND_NAMES+random.sample(train_able_pool,sample_num)
    server_status.ROUND_NAMES = [int(item) for item in server_status.ROUND_NAMES]

    
def aggregate_core(main_model,sub_model,coe=0.5)->nn.Module:
    
    result_model = Model()
    dictKeys = result_model.state_dict().keys()
    state_dict = {}
    for key in dictKeys:    
        state_dict[key] = main_model.state_dict()[key]*(1-coe)+sub_model.state_dict()[key]*coe
    result_model.load_state_dict(state_dict)
    return result_model

def init_round_names_core(server_status:Server_Status):
    #print(server_status.DATA_NAMES)
    server_status.ROUND_NAMES = random.sample(server_status.DATA_NAMES,server_status.PARALLEL_NUM)
    server_status.ROUND_NAMES = [int(item) for item in server_status.ROUND_NAMES]
    print("init round names is ",server_status.ROUND_NAMES)

def generate_mask_core(server_status:Server_Status,num_masks=10,mask_percentage=0.9):
    model:nn.Module = pickle.loads(server_status.MODEL_ENCODE)
    param = model.state_dict()
    masks = []
    for _ in range(num_masks):
        mask = {}
        for name in param.keys():
            mask[name] = torch.ones(param[name].shape)
            if "weight" in name:
                num_channels = param[name].shape[0]
                num_channels_to_mask = int(num_channels * mask_percentage)
                random_indices = random.sample(range(num_channels), num_channels_to_mask)
                for index in random_indices:
                    param[name][index] = torch.zeros(param[name][index].shape)
        masks.append(mask)
    server_status.MASKS = pickle.dumps(masks)



def init_server(args):
    print("init")
    server_status = Server_Status()
    init_log_core(args.logname)
    init_info_core(args.info,server_status)
    init_parallel_number_core(server_status,args)
    init_names_core(server_status,args)
    init_model_core(server_status)
    init_round_names_core(server_status)
    generate_mask_core(server_status)
    init_model_save(server_status,args)
    init_client_names(server_status,args)
    init_test_flie(server_status,args)


#########################################################
#####             Framework experiment tools        #####
#########################################################
from pre_dataset import CustomDataset
import datetime

def test(server_status:Server_Status):
    print("test begin")
    model = pickle.loads(server_status.MODEL_ENCODE)
    loader = test_loader_build_core(server_status.TEST_DATA_PATH)
    acc = test_core(model,loader,server_status.CUDA)
    test_log_core(acc,server_status)
        # 获取当前时间
#     current_time = datetime.datetime.now()

#     # 格式化时间并构建文件名
#     file_name = "{}_{:02d}_{:02d}_{:02d}_{:02d}_{}.pth".format(
#         current_time.month,
#         current_time.day,
#         current_time.hour,
#         current_time.minute,
#         current_time.second,
#         acc
# )
#     torch.save(model.state_dict(), os.path.join(server_status.MODEL_SAVE, file_name))

    
def test_loader_build_core(path):
    dataset_name = "test"
    custom_dataset = CustomDataset(dataset_name=dataset_name,cuda=server_status.CUDA,test_flie = path)
    loader = DataLoader(dataset=custom_dataset, batch_size=256, shuffle=True)
    return loader

def test_core(net:nn.Module,test_loader,cuda):
    
    net = copy.deepcopy(net)
    net.eval()  
    net = net.cuda(cuda)
    
    acc = 0

    for batch_num, (image, label) in enumerate(test_loader):
        image = image.cuda()
        label = label.cuda()
        output = net(image)
        acc += compute_accuracy(output, label)
    acc /= (batch_num + 1)  # 计算准确率
    #print(acc)
    return acc

def test_log_core(acc,server_status:Server_Status):
    testLog = logging.getLogger('test')
    server_status.CURR_ACC.append(acc)
    if acc>server_status.MAX_ACC:
        server_status.MAX_ACC = acc
        model = pickle.loads(server_status.MODEL_ENCODE)
        current_time = datetime.datetime.now()
        # 格式化时间并构建文件名
        file_name = "{}_{:02d}_{:02d}_{:02d}_{:02d}_{}.pth".format(
            current_time.month,
            current_time.day,
            current_time.hour,
            current_time.minute,
            current_time.second,
            acc
    )
        torch.save(model.state_dict(), os.path.join(server_status.MODEL_SAVE, file_name))
    acc_result = sum(server_status.CURR_ACC)/len(server_status.CURR_ACC)
    testLog.error("Round: {:d}, Max accuracy: {:.2f}%, Current accuracy: {:.2f}%".format(
        server_status.ROUND, server_status.MAX_ACC*100, acc_result*100
    ))
    
    
    

import torch
def compute_accuracy(possibility, label):
    sample_num = label.size(0)
    _, index = torch.max(possibility, 1)
    correct_num = torch.sum(label == index)
    return (correct_num/sample_num).item()

import os
def check_network():
    while(True):
        time.sleep(0.01)
        if Server_Status.RECV == 0:
            
            Server_Status.RECV,Server_Status.SENT = check_network_core()
            if os.path.exists("log_files/sent_log.txt"):
                os.remove("log_files/sent_log.txt")
            if os.path.exists("log_files/recv_log.txt"):
                os.remove("log_files/recv_log.txt")
        else:
            recv,sent = check_network_core(Server_Status.RECV,Server_Status.SENT)
            with open("log_files/sent_log.txt", "a") as f:
                f.write(f"{sent}\n")
            with open("log_files/recv_log.txt", "a") as f:
                f.write(f"{recv}\n")
import psutil
def check_network_core(begin_recv=0,begin_sent=0):

    current_bytes_sent = psutil.net_io_counters().bytes_sent - begin_sent
    current_bytes_recv = psutil.net_io_counters().bytes_recv - begin_recv
    
    return current_bytes_recv,current_bytes_sent
                
#########################################################
#####        Framework experiment tools end         #####
#########################################################
    

