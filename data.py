from torch.utils.data import Dataset
import torch
import numpy as np
import struct
from random import random

class EMNIST_Dataset_filter(Dataset):
    def __init__(self,data_path,del_per=0) -> None:
        super().__init__()
        self.del_per = del_per
        self.all_data = self.read_data(data_path = data_path)
        
            
            
    def read_data(self, data_path):
        data = unpickle(data_path)
        return data
    
    def __getitem__(self, key):
        data = self.all_data[key]
        label = data["label"]
        image = data["data"]
        image = torch.from_numpy(np.asarray(image)).reshape(3,32,32)
        return image,label
    
    def __len__(self)->int:
        return len(self.all_data.keys())

class Labels_Dataset_filter(Dataset):
    def __init__(self,data_path,select) -> None:
        super().__init__()
        self.select = select
        self.all_data = self.read_data(data_path = data_path,select=select)
        
    def read_data(self, data_path,select):
        data = unpickle(data_path)
        result = {}
        count = 0
        for key in data.keys():
            if data[key]["label"] in select:
                result[count] = data[key]
                count+=1
        return result
    
    def __getitem__(self, key):
        data = self.all_data[key]
        label = data["label"]
        image = data["data"]
        image = torch.from_numpy(np.asarray(image)).reshape(3,32,32)
        return image,label
    
    def __len__(self)->int:
        return len(self.all_data.keys())

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
