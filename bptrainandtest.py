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
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset

# 每次学习用的样本
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
class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel,self).__init__()
        self.fc1=nn.Linear(1*28*28,28)
        self.fc2=nn.Linear(28,10)

    def forward(self,input):
        x = input.view([input.size(0),1*28*28])    # 输入图像形状确定
        x = self.fc1(x)     # 全连接
        x = F.relu(x)   # 激活函数
        out = self.fc2(x)   # 输出层网络
        return F.log_softmax(out,dim=-1)

# 训练
Mymodel=MnistModel()
opt=Adam(Mymodel.parameters(),lr=0.001)



def train(epoch):
    data_loader=get_dataloader()
    for idx,(input,target) in enumerate(data_loader):
        opt.zero_grad()
        output=Mymodel(input)
        loss=F.nll_loss(output,target)
        loss.backward()
        opt.step()
        if(idx % 100==0):
            print(epoch,idx,loss.item())
# 模型的保存
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
        print("模型已存在，导入成功")
        test()
    else:
        print("开始训练")
        for i in range(100):
            train(i)

    # loader = get_dataloader(train=False)
    # for input,lable in loader:
    #     print(lable.size())
    #     break

    testdata=cv2.imread("D:/pythonItem/testdata.jpg", 0)
    # cv2.imshow("titel",testdata)
    # cv2.waitKey(0)

    transf = transforms.ToTensor()
    img_tensor = transf(testdata)  # tensor数据格式是torch(C,H,W)
    # print(img_tensor.size())
    # print(img_tensor[0].numpy())

    # plt.matshow(img_tensor[0].numpy(), cmap=plt.get_cmap('Greys'), alpha=1)  # , alpha=0.3
    # plt.show()

    TargetOutput=Mymodel(img_tensor)
    TargetPre = TargetOutput.max(dim=-1)[-1]
    print("识别为:",TargetPre[0].numpy())

