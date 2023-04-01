#数据集准备(dataLoading)
import os.path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from  torchvision.transforms import Compose,ToTensor,Normalize
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import cv2
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt

BATCH_SIZE=128
TEST_BATCH_SIZE=1000

#准备数据集
def get_dataloader(train=True):
    transform_fn=Compose([
        ToTensor(),
        Normalize(mean=(0.13,),std=(0.30,))
    ])
    dataset=MNIST(root="./data",train=True,transform=transform_fn)
    data_loader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)
    return data_loader

# 建立模型

import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.fc1 = nn.Linear(32*7*7, 10)

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128* 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 128*7*7)
        x = self.fc1(x)
        return x

# 训练
# 创建ConvNet实例和优化器
Mymodel=ConvNet()
opt=optim.SGD(Mymodel.parameters(),lr=0.001)

# 开始训练
def train(epoch):
    data_loader=get_dataloader()
    for idx, (images, labels) in enumerate(data_loader):
        # 前向传播
        outputs = Mymodel(images)
        loss = nn.functional.cross_entropy(outputs, labels)

        # 反向传播和优化
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (idx % 100 == 0):
            print(epoch, idx, loss.item())
        if (idx % 100 ==0):
            torch.save(Mymodel.state_dict(),"./saved_model/MyModel.pkl")
            torch.save(opt.state_dict(),"./saved_model/Myopt.pkl")



def get_dataloader(train=True, batch_size=128):
    transform_fn = Compose([
        ToTensor(),
        Normalize(mean=(0.13,), std=(0.30,))
    ])
    dataset = MNIST(root="./data", train=train, transform=transform_fn)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


def test():
    loss_list=[]
    acc_list=[]
    test_dataloader=get_dataloader(train=False,batch_size=TEST_BATCH_SIZE)
    for idx,(input,target) in enumerate(test_dataloader):
        with torch.no_grad():
            output = Mymodel(input)
            cur_loss=F.nll_loss(output,target)
            loss_list.append(cur_loss)
#             计算准确率
            pred=output.max(dim=-1)[-1]
            cur_acc=pred.eq(target).float().mean()
            acc_list.append(cur_acc)
    print("平均准确率，平均损失：",np.mean(acc_list),np.mean(loss_list))

if __name__=="__main__":
    if os.path.exists("./saved_model/MyModel.pkl"):
        Mymodel.load_state_dict(torch.load("./saved_model/MyModel.pkl"))
        opt.load_state_dict(torch.load("./saved_model/Myopt.pkl"))
        # torch.save(Mymodel.state_dict(), "./saved_model/MyModel.pt")
        # torch.save(opt.state_dict(), "./saved_model/Myopt.pt")
        print("模型已存在，导入成功")
    else:
        print("开始训练")
        # cpu_num是一个整数
        torch.set_num_threads(8)
        for i in range(200):
            train(i)

    # loader = get_dataloader(train=False)
    # for input,lable in loader:
    #     print(lable.size())
    #     break

    testdata=cv2.imread("D:/pythonItem/testdata.jpg", 0)
    # cv2.imshow("titel",testdata)
    # cv2.waitKey(0)

    transf = transforms.ToTensor()
    img_tensor = transf(testdata)


    # tensor数据格式是torch(C,H,W)
    # print(img_tensor.size())
    # print(img_tensor[0].numpy())

    # plt.matshow(img_tensor[0].numpy(), cmap=plt.get_cmap('Greys'), alpha=1)  # , alpha=0.3
    # plt.show()

    TargetOutput=Mymodel(img_tensor)
    TargetPre = TargetOutput.max(dim=-1)[-1]
    print("识别为:",TargetPre[0].numpy())

