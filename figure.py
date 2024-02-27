import os
import shutil
from utils import *
from PIL import Image
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        "cifar10", "/home/dell/xy/new/mobicom-TWT/cifar/data", "./logs/", "noniid", 10, beta=0.1)

import matplotlib.pyplot as plt
import numpy as np

def plot_heatmap(traindata_cls_counts):
    # 获取所有party的编号
    parties = list(traindata_cls_counts.keys())
    
    # 获取所有类别的编号（0-199）
    all_classes = list(range(10))

    # 初始化一个二维数组，用于存储每个party对应每个类别的数据数量
    data_counts = np.zeros((len(all_classes), len(parties)))

    # 将字典中的数据填充到数组中
    for party, class_counts in traindata_cls_counts.items():
        for class_id, count in class_counts.items():
            data_counts[class_id, party] = count

    # 获取反转的颜色映射
    cmap_reversed = plt.cm.get_cmap('viridis').reversed()

    # 绘制热图，使用反转的颜色映射
    plt.imshow(data_counts, cmap=cmap_reversed, interpolation='nearest', aspect='auto')
    plt.colorbar(label='Data Count', orientation='vertical')

    # 设置横坐标和纵坐标刻度
    plt.xticks(range(len(parties)), parties)
    y_interval = 11
    plt.yticks(range(0, len(all_classes), y_interval), all_classes[::y_interval])
    
    # 设置坐标轴标签
    plt.xlabel('Party')
    plt.ylabel('Class')
    
    # 设置图标题
    plt.title('Data Count per Class and Party')

    # 保存图片到指定路径
 

    # 显示图形
    plt.show()


# 使用函数绘制热图
plot_heatmap(traindata_cls_counts)
plt.savefig('/home/dell/xy/A/beta_0.1.png', bbox_inches='tight')


# for i in range(100):
#     print(len(traindata_cls_counts))

# def plot_net_data_counts(net_cls_counts, save_path=None):
#     # 获取所有party的编号
#     parties = list(net_cls_counts.keys())
    
#     # 获取所有类别的编号
#     all_classes = set(class_id for counts in net_cls_counts.values() for class_id in counts.keys())

#     # 初始化一个二维数组，用于存储每个party对应每个类别的数据数量
#     data_counts = [[net_cls_counts[party].get(class_id, 0) for party in parties] for class_id in all_classes]

#     # 转换为NumPy数组
#     data_counts = np.array(data_counts)

#     # 绘制热图
#     plt.imshow(data_counts, cmap='viridis', interpolation='nearest')
#     plt.colorbar(label='Data Count')
    
#     # 设置横坐标和纵坐标刻度
#     plt.yticks(range(len(all_classes)), all_classes)
#     plt.xticks(range(len(parties)), parties)
    
#     # 设置坐标轴标签
#     plt.ylabel('Class')
#     plt.xlabel('Party')
    
#     # 设置图标题
#     plt.title('Data Count per Class and Party')

#     # 保存图片
#     if save_path:
#         plt.savefig(save_path, bbox_inches='tight')

#     # 显示图形
#     plt.show()
# # 保存图片到指定路径
# plot_net_data_counts(traindata_cls_counts, save_path='/home/dell/xy/mobicom-TWT/tiny-image/xy_dataset/2.png')
