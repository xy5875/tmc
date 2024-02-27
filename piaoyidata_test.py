import torch
from torchvision import datasets, transforms
import os
from torchvision.utils import save_image

# 定义转换
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 下载 CIFAR-10 测试集
test_dataset = datasets.CIFAR10(root='/home/dell/xy/AFLvsGFL/data/cifar', train=False, download=False, transform=transform)

# 定义存储图像的主文件夹
base_folder = "/home/dell/xy/AFLvsGFL/data/test"
os.makedirs(base_folder, exist_ok=True)

# 定义 5 个客户端和对应的标签
client_labels = {
    0: [0, 8],
    1: [1, 3],
    2: [2, 6],
    3: [4, 5],
    4: [7, 9],
}

# 定义每个客户端的图像数量
images_per_client = 1000

# 使用一个集合来跟踪已选择的图像索引，确保在所有客户端中不选择重复的图像
selected_images = set()

# 为每个客户端创建文件夹
for client_id, labels in client_labels.items():
    client_folder = os.path.join(base_folder, f"{client_id}")
    os.makedirs(client_folder, exist_ok=True)

    # 为每个标签创建文件夹，并保存图像
    for label in labels:
        label_folder = os.path.join(client_folder, str(label))
        os.makedirs(label_folder, exist_ok=True)

        # 选择该客户端对应的图像，确保没有重复
        client_images = set()
        while len(client_images) < images_per_client:
            i = torch.randint(0, len(test_dataset), size=(1,)).item()

            # 确保选择的图像对应于指定的标签，并且它不在已经选择的图像集合中
            if i not in selected_images:
                image, target = test_dataset[i]
                if target == label:
                    selected_images.add(i)
                    client_images.add(i)

                    # 保存图像
                    image_path = os.path.join(label_folder, f"{len(client_images) - 1}.png")
                    save_image(image, image_path)
