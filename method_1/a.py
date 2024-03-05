import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from model import ResNet34 as Model

# class TrainDataset(Dataset):
#     class_shot = None
#     def __init__(self, dataset_path, list=[5], cuda=1):
#         self.root_folder = dataset_path
#         self.transform = transforms.Compose([
#             transforms.ToTensor()
#         ])
#         self.cuda = cuda

#         self.data = None
#         # Randomly choose class_shot once during initialization
#         if TrainDataset.class_shot is None:
#             labels = [int(label) for label in os.listdir(self.root_folder)]
#             labels_in_listA = [label for label in labels if label in list]
    
#             if labels_in_listA:
#                 TrainDataset.class_shot = random.choice(labels_in_listA)
#             else:
#                 print("listA 中的元素都不在 labels 中，无法选择在 listA 中的元素。")
#         self.load_data()

#     def load_data(self):
#         labels = [int(label) for label in os.listdir(self.root_folder)]
#         print("shot is ", TrainDataset.class_shot)
#         positive_samples = []
#         negative_samples = []
#         # Load positive samples (label = 0)
#         for label in labels:
#             label_folder = os.path.join(self.root_folder, str(label))
#             image_list = os.listdir(label_folder)

#             for img_name in image_list:
#                 image_path = os.path.join(label_folder, img_name)
#                 image = Image.open(image_path).convert('RGB')

#                 if self.transform:
#                     image = self.transform(image)

#                 if label == TrainDataset.class_shot:
#                     label_tensor = torch.tensor(0)
#                     positive_samples.append((image, label_tensor))

#         # Check if the number of positive samples is 1
#         if len(positive_samples) == 1:
#             # Duplicate the positive sample 10 times
#             positive_samples *= 10

#         # Load negative samples (label = 1)
#         for label in labels:
#             label_folder = os.path.join(self.root_folder, str(label))
#             image_list = os.listdir(label_folder)

#             for img_name in image_list:
#                 image_path = os.path.join(label_folder, img_name)
#                 image = Image.open(image_path).convert('RGB')

#                 if self.transform:
#                     image = self.transform(image)

#                 if label != TrainDataset.class_shot:
#                     label_tensor = torch.tensor(1)
#                     negative_samples.append((image, label_tensor))

#         # Randomly select the same number of negative samples
#         num_positive_samples = len(positive_samples)
#         num_negative_samples = len(negative_samples)
        
#         # Repeat sampling from negative samples until it meets the requirement
#         while num_negative_samples < num_positive_samples:
#             negative_samples.extend(random.choices(negative_samples, k=num_positive_samples - num_negative_samples))
#             num_negative_samples = len(negative_samples)

#         # Randomly select the same number of negative samples
#         negative_samples = random.sample(negative_samples, num_positive_samples)

#         # Combine positive and negative samples
#         data = positive_samples + negative_samples
#         random.shuffle(data)

#         if self.cuda:
#             data = [(image.cuda(self.cuda), label_tensor.cuda(self.cuda)) for image, label_tensor in data]

#         self.data = data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         return self.data[index]


# # Example usage:
# dataset_path = '/home/dell/xy/TMC/data/train/4'
# dataset = TrainDataset(dataset_path=dataset_path) 
# class_shot = dataset.class_shot
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)



import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # 遍历数据集文件夹，收集图像和标签
        for label in os.listdir(self.root_dir):
            label_dir = os.path.join(self.root_dir, label)
            for image_name in os.listdir(label_dir):
                self.images.append(os.path.join(label_dir, image_name))
                self.labels.append(int(label))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# 设置数据集根目录和数据转换
data_root = '/home/dell/xy/TMC/data/xy'
transform = transforms.Compose([
    
    transforms.ToTensor()
     # 假设这是你的数据归一化
])

# 创建自定义数据集
custom_dataset = CustomDataset(root_dir=data_root, transform=transform)

# 创建数据加载器
batch_size = 32
dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)




from pre_dataset import CustomDataset
dataset_name = "test"
custom_dataset = CustomDataset(dataset_name=dataset_name,cuda=0,test_flie = "/home/dell/xy/TMC/data/test/5")
test_loader = DataLoader(dataset=custom_dataset, batch_size=32, shuffle=True)
    
    
    
    
    
import os
import shutil

# 假设你有一个名为dataloader的数据加载器
# 假设你有一个名为data_dir的文件夹，其中包含子文件夹0、1、2等，用于存储不同类别的图像

# import os
# import torchvision.transforms.functional as TF
# from PIL import Image

# # 假设你有一个名为dataloader的数据加载器
# # 假设你有一个名为data_dir的文件夹，其中包含子文件夹0、1、2等，用于存储不同类别的图像
# j = 0
# for batch_idx, (images, labels) in enumerate(dataloader):
#     print("label ",labels)
#     for image, label in zip(images, labels):
        
#         # 获取当前图像的标签
#         label = label.item()
        
#         # 构建目标文件夹的路径
#         target_folder = os.path.join('/home/dell/xy/TMC/data/xy', str(label))
        
#         # 确保目标文件夹存在，如果不存在，则创建它
#         if not os.path.exists(target_folder):
#             os.makedirs(target_folder)
        
#         # 将PyTorch张量转换为PIL图像对象
#         image_pil = TF.to_pil_image(image)
        
#         # 构建目标图像文件的路径
#         image_filename = f'image_{batch_idx}_{j}.png'
#         target_image_path = os.path.join(target_folder, image_filename)
        
#         # 保存图像到目标路径
#         image_pil.save(target_image_path)
#         j = j+1






import torch.nn as nn
import torch.optim as optim
net = Model()
net = net.cuda(0)
cross_entropy = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr = 0.01)

acc=[]

def compute_accuracy(output, label):
    predictions = output.argmax(dim=1)
    correct = (predictions == label).float()
    # True Positive (TP): Prediction is positive and label is also positive
    TP = ((predictions == 0) & (label == 0)).sum().item()
    # True Negative (TN): Prediction is negative and label is also negative
    TN = ((predictions == 1) & (label == 1)).sum().item()
    # False Positive (FP): Prediction is positive but label is negative
    FP = ((predictions == 0) & (label == 1)).sum().item()
    # False Negative (FN): Prediction is negative but label is positive
    FN = ((predictions == 1) & (label == 0)).sum().item()

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return accuracy

def test_core(net, test_loader, cuda):
    net.eval()  
    net = net.cuda(cuda)
    
    acc = 0

    for batch_num, (image, label) in enumerate(test_loader):
        image = image.cuda()
        label = label.cuda()
        output = net(image)
        acc += compute_accuracy(output, label)
    acc /= (batch_num + 1)  # 计算精度
    return acc


for epoch in range(2000): 
    net = net.train()           
    for batch_num,(image,label) in enumerate(dataloader):
       
        image = image.cuda(0)
        label = label.cuda(0)
        output = net(image)
        entropy_num = cross_entropy(output,label)
        loss = entropy_num
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
            
    test_accuracy = test_core(net, test_loader,0)
    acc.append(test_accuracy)
    with open('center.txt', 'w') as file:
    # 将变量写入文件
        file.write("epoch =  {}\n".format(epoch))
        file.write("acc =  {}\n".format(acc))
        #file.write("Delay is : {}\n".format(server_status.delay[int(server_status.NAME)]))
        











#初始化计数器
# print("class_shot is ",class_shot)
# count_0 = 0
# count_1 = 0

# for batch_idx, (images, labels) in enumerate(dataloader):
#     # 统计每个批次中标签为 0 和 1 的数量
#     print("label is ",labels)
#     count_0 += (labels == 0).sum().item()
#     count_1 += (labels == 1).sum().item()

# # 打印结果
# print(f"Number of label 0: {count_0}")
# print(f"Number of label 1: {count_1}")

# # 统计原数据集的类别和数目
# original_dataset_path = '/home/dell/xy/TMC/data/train/0'
# original_labels_count = {}

# # 遍历原数据集文件夹
# for label in os.listdir(original_dataset_path):
#     label_folder = os.path.join(original_dataset_path, label)
#     # 统计每个类别的样本数量
#     original_labels_count[label] = len(os.listdir(label_folder))

# # 打印原数据集的类别和数量
# print("Original dataset labels and counts:")
# for label, count in original_labels_count.items():
#     print(f"Label {label}: {count} samples")
# acc = [0.0, 17.397836595773697, 100.0, 100.0, 80.0, 80.0, 100.0, 100.0, 80.13671875, 79.98046875, 40.01953125, 84.88131061196327, 85.2073322236538, 99.03545677661896, 59.20072134584189, 92.12740406394005, 82.50600978732109, 53.71995210647583, 35.00000022351742, 75.19982010126114, 78.83263252675533, 87.7659259736538, 88.64483207464218, 63.6237982660532, 52.37379841506481, 73.50510843098164, 63.0874402821064, 72.12289683520794, 31.61358185112476, 71.87049321830273, 61.87500037252903, 77.1259018778801, 85.78725978732109, 67.00871415436268, 47.02524058520794, 52.69531274214388, 86.26502461731434, 49.07902657985687, 39.827223755419254, 45.0646036490798, 25.47325733117759, 38.42548094689846, 85.12319758534431, 23.879206916317344, 27.844050684943795, 46.26051694620401, 82.16947130858898, 31.132812686264515, 36.74579340964556, 29.278846308588978, 19.64693522080779, 49.55228392034769, 31.41225978732109, 72.82902657985687, 68.49459178745747, 76.08924306929111, 78.33834171295166, 28.123497813940045, 47.12289687246084, 61.35066136717796, 91.09976008534431, 83.50060142576694, 41.416767090559006, 61.699219048023224, 65.05558922886848, 46.62409879267216, 21.460336707532406, 59.25781276077032, 26.89603380858898, 42.96574544161558, 44.2833536490798, 81.74729570746422, 80.89543279260397, 98.45552921295166, 56.89302906394005, 76.48587763309479, 83.95432710647583, 76.87950730323792, 52.19351001083851, 74.00691147893667, 61.20042096823454, 98.10997605323792, 62.30919498950242, 22.325721234083176, 70.54086573421955, 61.32512051612139, 73.24218779802322, 34.07451940700412, 69.0294473618269, 64.95042093098164, 71.74729615449905, 91.57301723957062, 90.65204367041588, 62.2521036863327, 64.46213975548744, 86.38221189379692, 96.6271036863327, 86.98317348957062, 57.52103388309479, 65.57542085647583, 39.667968936264515, 28.96484386175871, 81.846454590559, 89.55679133534431, 34.101562555879354, 29.813702069222924, 22.3001803830266, 49.35997627675533, 37.01772850006819, 33.27974773943424, 31.947115547955036, 57.84405078738928, 61.63461573421956, 95.713642090559, 42.770432718098164, 46.42427906394005, 63.03185127675533, 81.40174314379692, 40.211839117109776, 76.182392090559, 66.98467573150992, 53.34134638309479, 34.25480796024203, 36.06370208784938, 40.4912861995399, 45.47626230865717, 41.7082332726568, 21.038161143660545, 45.695613250136375, 61.021634861826904, 21.9365986995399, 26.397235803306103, 14.498197175562384, 94.67097401618958, 62.343750186264515, 78.04988026618958, 96.90655082464218, 96.94110617041588, 96.94561332464218, 63.879206851124756, 61.0411661118269, 58.7905652076006, 79.78816136717796, 87.23407492041588, 89.9429090321064, 56.17638241499663, 58.79507238045335, 77.30468787252903, 34.17968761175871, 33.716947212815285, 57.372296061366804, 79.84825748950243, 7.764423126354814, 98.93179103732109, 46.0096153896302, 32.15745199471712, 69.40354600548744, 19.97896640561521, 20.564903914928436, 44.44110594689846, 41.36568530462682, 59.7776444721967, 49.68599781394005, 71.18689939379692, 39.33593764901161, 45.2403848618269, 30.788762159645554, 56.25150263309479, 52.77043290436267, 42.90865398943424]
# print("len is ",len(acc))
# fisher = [0.0, 0.0, -28.37560460716486, -61.66031799465418, -61.66031799465418, -169.5115319734905, -169.5115319734905, -201.7106224691961, -303.31406133418204, -414.11064331192756, -442.96670225221897, -549.1081341286772, -642.7297505144088, -752.1369877714606, -778.4069861654134, -811.7771130796464, -839.3577498193772, -862.8426613748015, -896.5148955970944, -920.0267018802406, -949.3396159462573, -1030.5294117421436, -1129.5753093656385, -1158.679994906881, -1158.679994906881, -1244.8317540534917, -1277.037360467497, -1377.204343979698, -1407.8882536343153, -1436.6174991517182, -1436.6174991517182, -1457.484297468778, -1477.8740066620376, -1550.6746569330717, -1550.6746569330717, -1580.1825101477007, -1597.870082011461, -1611.8086192151677, -1682.7285731171432, -1709.99060116705, -1731.6804540990524, -1731.6804540990524, -1808.8468311779325, -1883.5089176626243, -1913.006310122164, -1926.2577832366405, -1955.436161040754, -1955.436161040754, -1966.72426338257, -1986.4930518180663, -2000.4629849890662, -2000.4629849890662, -2081.3822990255685, -2082.7801076534024, -2102.401403957797, -2102.401403957797, -2102.401403957797, -2174.4613344951113, -2199.077392541575, -2229.6301677848724, -2229.6301677848724, -2251.024101755819, -2284.281179479095, -2314.0690635365945, -2360.378778075935, -2388.9617673867806, -2394.7823038170072, -2453.089806065699, -2453.498013903491, -2464.132805005984, -2526.751482324819, -2572.789840665297, -2572.789840665297, -2582.9780764797474, -2582.9780764797474, -2599.2140704041817, -2653.598669143845, -2653.598669143845, -2688.1676308277715, -2689.0039729669243, -2689.3417476539016, -2712.415585634064, -2729.7719163411657, -2753.681532238914, -2760.310504021718, -2767.3462879569397, -2770.4282937997614, -2770.4282937997614, -2803.403505200913, -2831.1145775363593, -2831.322665798489, -2847.7776754715564, -2847.7776754715564, -2847.7776754715564, -2847.7776754715564, -2858.5493979921125, -2869.9925484988294, -2904.700415968468, -2917.8782738575633, -2925.5133513037385, -2945.6850780027257, -2945.6850780027257, -2961.428967831171, -2972.4897384206365, -2972.4897384206365, -2972.4897384206365, -2972.4897384206365, -2972.4897384206365, -2983.065403884912, -2983.065403884912, -3000.4598761109723, -3020.4490702151584, -3020.4490702151584, -3054.781339964541, -3091.314877237121, -3091.314877237121, -3095.7121908099402, -3108.399537133799, -3114.64987091261, -3122.749703073845, -3122.749703073845, -3122.749703073845, -3142.481634279684, -3142.481634279684, -3157.1980039516493, -3157.1980039516493, -3190.7623424342996, -3220.638694669043, -3235.576123231421, -3235.576123231421, -3235.576123231421, -3242.7505246632913, -3242.7505246632913, -3242.7505246632913, -3242.7505246632913, -3242.7505246632913, -3275.665019476711, -3286.635216251639, -3286.635216251639, -3286.635216251639, -3293.430959560222, -3308.5826797737514, -3332.126171069668, -3362.1750060953027, -3363.7709029454372, -3363.7709029454372, -3386.0176993751893, -3392.4976884347075, -3418.2515018903455, -3420.590599178375, -3440.1608209705464, -3468.3217152415327, -3496.083395705781, -3496.175201376237, -3512.277461736351, -3512.277461736351, -3512.277461736351, -3519.7872853280905, -3519.7872853280905, -3519.7872853280905, -3532.4229101999126, -3575.9585838787652, -3582.150300926737, -3597.221640925108, -3598.899991517024, -3607.9265291217266, -3611.0485058611152, -3611.0485058611152, -3614.4095972990976, -3634.6021560003933, -3638.579341603959, -3638.579341603959, -3648.3030882847247, -3648.8018723454697, -3660.902309443296, -3678.0858641760024, -3694.889477251407, -3700.4907893341037, -3708.1663853418027, -3708.1663853418027, -3708.1663853418027, -3711.912651448892, -3730.28729096643, -3753.4810888296643, -3767.220271656058, -3768.9575690928477, -3786.054441927827, -3812.4152995074514]
# print("fisher is ",len(fisher))