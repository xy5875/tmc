import torch
from torchvision import datasets, transforms
import os
from PIL import Image
from torchvision.utils import save_image

# 定义转换
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 下载CIFAR-10数据集
train_dataset = datasets.CIFAR10(root='/home/dell/xy/new/mobicom-TWT/cifar/data', train=True, download=False, transform=transform)

# 定义存储图像的主文件夹
base_folder = "/home/dell/xy/new/mobicom-TWT/cifar/new_iid_data"
os.makedirs(base_folder, exist_ok=True)

# 定义10个client
num_clients = 10
images_per_client = 500

# 使用一个集合来跟踪已选择的图像索引，确保在所有客户端中不选择重复的图像
selected_images = set()

# 为每个client创建文件夹
for client_id in range(num_clients):
    client_folder = os.path.join(base_folder, f"{client_id}")
    os.makedirs(client_folder, exist_ok=True)

    # 为每个label创建文件夹，并保存图像
    for label in range(10):
        label_folder = os.path.join(client_folder, str(label))
        os.makedirs(label_folder, exist_ok=True)

        # 选择该client对应的图像，确保没有重复
        client_images = set()
        while len(client_images) < images_per_client:
            i = torch.randint(0, len(train_dataset), size=(1,)).item()

            # 确保选择的图像对应于指定的标签，并且它不在已经选择的图像集合中
            if i not in selected_images:
                image, target = train_dataset[i]
                if target == label:
                    selected_images.add(i)
                    client_images.add(i)

                    # 保存图像
                    image_path = os.path.join(label_folder, f"{len(client_images) - 1}.png")
                    save_image(image, image_path)


