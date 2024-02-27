import os
import shutil
#from dataset import *
from PIL import Image
import torchvision.transforms.functional as TF
from utils import *

# clinet_index,traindata_cls_counts = partition_data( datadir='/home/dell/xy/mobicom-TWT/tiny-image/tiny_imagenet/tiny-imagenet-200', n_parties=10, beta=0.1)
# test_dataloader = get_test_dataloader(datadir='/home/dell/xy/mobicom-TWT/tiny-image/tiny_imagenet/tiny-imagenet-200', test_bs=1 )
# X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
#         "cifar10", "/home/dell/xy/new/mobicom-TWT/ImageNet/cifar/data", "./logs/", "noniid", 10, beta=0.5)
# dataidxs = net_dataidx_map[i]    
# train_dl_local, test_dl_local, _, _ = get_dataloader('cifar10',"/home/dell/xy/new/mobicom-TWT/ImageNet/cifar/data", 1, 1, dataidxs)


main_folder = "/home/dell/xy/new/mobicom-TWT/ImageNet/cifar/test_data"
labels_set = set()
for i in range(1):
    labels_set.clear()
    print("i is ",i)
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        "cifar10", "/home/dell/xy/new/mobicom-TWT/ImageNet/cifar/data", "./logs/", "noniid", 10, beta=0.5)
    dataidxs = net_dataidx_map[i]    
    train_dl_local, test_dl_local, _, _ = get_dataloader('cifar10',"/home/dell/xy/new/mobicom-TWT/ImageNet/cifar/data", 1, 1, dataidxs)
    id_folder = os.path.join(main_folder, str(i))
    os.makedirs(id_folder, exist_ok=True)
    
# 假设train_dataloader是一个包含图片和标签的列表
    for image, label in test_dl_local:
        # 将标签加入集合
        labels_set.add(label)

# 创建子文件夹
    #print("lable set  is ",labels_set)
    for label in labels_set:
        #print("lable is ",label)
        label_folder = os.path.join(id_folder, str(label.item()))
        os.makedirs(label_folder, exist_ok=True)

    # 重新遍历train_dataloader，将每个样本的图片保存到相应标签的文件夹中
    t = 0
    for image, label in test_dl_local:
        label_folder = os.path.join(id_folder, str(label.item()))
        
        # 生成图片文件名（可以根据需要修改）
        image_filename = f"image_{t}.png"  # 请替换unique_identifier为实际的唯一标识符

        
#         transform_reverse = transforms.Compose([
#     transforms.Normalize(mean=(-1, -1, -1), std=(2, 2, 2)),  # 逆归一化
#     transforms.ToPILImage(),  # 转换为PIL Image
# ])
        transform_reverse = transforms.Compose([
    transforms.ToPILImage(),  # 转换为PIL Image
])
        original_image = transform_reverse(image.squeeze())
        # 完整的图片路径
        image_path = os.path.join(label_folder, image_filename)
        
        original_image.save(image_path)

        # 保存图片
        #shutil.copy(image, image_path)
        t+=1
