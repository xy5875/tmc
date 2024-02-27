import threading
from typing import Dict, List
from uuid import UUID
import torch.nn as nn
from collections import deque

CACHE_PATH = "server.json"

import threading

class Server_Status(object):
    _instance_lock = threading.Lock()
    client_config = {}
    SERVER_WEIGHT_PATH="./server_weight/"
    NETS = []
    DATA_NAMES = []
    ROUND_NAMES = []
    TRAINING_NAMES = []
    CUDA = 0
    PARALLEL_NUM = 5
    SEND_NUM = 0
    ROUND = 0
    MAX_ACC = 0
    MODEL_ENCODE = ""
    DATA_INFO = "./data/info.json"
    TEST_DATA_PATH = "data/data_public"
    CUDA_LIST = [0,0,0,1,1,1,1]
    ROUND = 0
    MASKS = ""
    CURR_ACC = deque(maxlen=5)
    MODEL_SAVE = ""
    
    def __init__(self):
        pass


    def __new__(cls, *args, **kwargs):
        if not hasattr(Server_Status, "_instance"):
            with Server_Status._instance_lock:
                if not hasattr(Server_Status, "_instance"):
                    Server_Status._instance = object.__new__(cls)  
        return Server_Status._instance
