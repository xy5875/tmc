from turtle import forward
import torch.nn as nn
import torch.nn.functional as F
import torch

class ResBlock(nn.Module):
    def __init__(self,inchannels,outchannels,stride=1):
        super().__init__()
        self.convBlock = nn.Sequential(
            nn.Conv2d(inchannels,outchannels,kernel_size=3,stride=stride,padding=1),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannels,outchannels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(outchannels)
        )
        self.shortcut = nn.Sequential()
        if inchannels!=outchannels or stride!=1:
            self.shortcut.add_module("shortcut",nn.Conv2d(inchannels,outchannels,kernel_size=1,stride=stride))
        
    def forward(self,x):
        out = self.convBlock(x)
        out += self.shortcut(x)
        return F.relu(out)
    
class DepthwiseBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=1):
        super().__init__()
        self.dw = nn.Conv2d(in_channels,in_channels,kernel_size=kernel_size,stride=stride,padding=padding,groups=in_channels)
        self.pw = nn.Conv2d(in_channels,out_channels,kernel_size=1)

    def forward(self,x):
        x1 = self.dw(x)
        x2 = F.relu(x1)
        x3 = self.pw(x2)
        x4 = F.relu(x3)
        return x4
        

class TwoNNNet(nn.Module):
    def __init__(self,numClass=10):
        super().__init__()
        self.pool4 = nn.AvgPool2d(kernel_size=4)
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.conv1 = DepthwiseBlock(in_channels=3,out_channels=64,kernel_size=3)
        self.conv1_1 = DepthwiseBlock(in_channels=64,out_channels=64,kernel_size=3)
        self.conv2 = DepthwiseBlock(in_channels=64,out_channels=128,kernel_size=3)
        self.conv2_2 = DepthwiseBlock(in_channels=128,out_channels=128,kernel_size=3)
        self.conv3 = DepthwiseBlock(in_channels=128,out_channels=256,kernel_size=3)
        self.conv3_3 = DepthwiseBlock(in_channels=256,out_channels=256,kernel_size=3)
        self.fc = nn.Linear(1024,numClass)
        
        
    def forward(self,x):
        x1 = self.conv1(x)
        x1 = self.conv1_1(x1)
        x2 = self.pool2(x1) # 16X16X64
        
        x3 = self.conv2(x2)
        x3 = self.conv2_2(x3) 
        x4 = self.pool2(x3) # 8X8X128
        
        x5 = self.conv3(x4)
        x6 = self.pool2(x5) # 4X4X256
        
        x7 = self.conv3_3(x6)
        x7 = self.pool2(x7) # 2X2X256
        
        x8 = x7.view(x7.size(0),-1)
        x9 = self.fc(x8)
        return x9
        


class ResNet34(nn.Module):
    def __init__(self,numClass=10):
        super().__init__()
        self.pool = nn.MaxPool2d(stride=2,kernel_size=3,padding=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.layer1 = ResBlock(64,64,1)
        self.layer2 = nn.Sequential(ResBlock(64,128,2),ResBlock(128,128,1))
        self.layer3 = nn.Sequential(ResBlock(128,256,2),ResBlock(256,256,1))
        self.layer4 = nn.Sequential(ResBlock(256,512,2),ResBlock(512,512,1))
        self.fc = nn.Linear(512,numClass)
        
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = F.avg_pool2d(x5,2)
        x7 = x6.view(x6.size(0),-1)
        output = self.fc(x7)
        return output
    
    
class ResNet_Simple(nn.Module):
    def __init__(self,numClass=10):
        super().__init__()
        self.pool = nn.MaxPool2d(stride=2,kernel_size=3,padding=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=3,stride=2,padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True))
        self.layer1 = ResBlock(16,16,1)
        # self.layer2 = nn.Sequential(ResBlock(4,8,2),ResBlock(8,8,1))
        # self.layer3 = nn.Sequential(ResBlock(8,16,2),ResBlock(16,16,1))
        # self.layer4 = nn.Sequential(ResBlock(16,32,2),ResBlock(32,32,1))
        self.layer2 = nn.Sequential(ResBlock(16,32,2),ResBlock(32,32,1))
        self.layer3 = nn.Sequential(ResBlock(32,64,2),ResBlock(64,64,1))
        self.layer4 = nn.Sequential(ResBlock(64,128,2),ResBlock(128,128,1))
        self.fc = nn.Linear(128,numClass)
        
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x6 = F.avg_pool2d(x5,2)
        x7 = x6.view(x6.size(0),-1)
        output = self.fc(x7)
        return output
        
        
            
class SimpleNet(nn.Module):
    def __init__(self,num_class) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.pool = nn.AvgPool2d(stride=2,kernel_size=3,padding=1)
        self.fc = nn.Linear(512,num_class)
        
        
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.pool(x1)
        x3 = self.conv2(x2)
        x4 = self.pool(x3)
        x5 = self.pool(x4)
        x6 = x5.view(x5.size(0),-1)
        x7 = self.fc(x6)
        F.softmax(x7, dim=1)
        return x7
        
        
# 建立一个四层感知机网络
class MLP(torch.nn.Module):  
    def __init__(self,num_class):
        super(MLP,self).__init__()    # 
        # 初始化三层神经网络 两个全连接的隐藏层，一个输出层
        self.fc1 = torch.nn.Linear(3072,512)  # 第一个隐含层  
        self.fc2 = torch.nn.Linear(512,128)  # 第二个隐含层
        self.fc3 = torch.nn.Linear(128,num_class)   # 输出层
        
    def forward(self,din):
        din = din.view(-1,32*32*3)       # 将一个多行的Tensor,拼接成一行
        dout = F.relu(self.fc1(din))   # 使用 relu 激活函数
        dout = F.relu(self.fc2(dout))
        dout = F.softmax(self.fc3(dout), dim=1)  # 输出层使用 softmax 激活函数
        # 10个数字实际上是10个类别，输出是概率分布，最后选取概率最大的作为预测值输出
        return dout

class SimpleCNN(nn.Module):
    def __init__(self, numClass=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 * 8 * 8, numClass)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
    
# class ResNet34(nn.Module):
#     def __init__(self, num_classes=10):
#         super(ResNet34, self).__init__()

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         self.layer1 = self.make_layer(64, 64, 3, stride=1)
#         self.layer2 = self.make_layer(64, 128, 4, stride=2)
#         self.layer3 = self.make_layer(128, 256, 6, stride=2)
#         self.layer4 = self.make_layer(256, 512, 3, stride=2)

#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(512, num_classes)

#     def make_layer(self, in_channels, out_channels, num_blocks, stride):
#         layers = [ResBlock(in_channels, out_channels, stride)]
#         for _ in range(1, num_blocks):
#             layers.append(ResBlock(out_channels, out_channels, 1))
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.pool(x)

#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)

#         x = self.avg_pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)

#         return x