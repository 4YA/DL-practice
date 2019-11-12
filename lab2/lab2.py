from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

import dataloader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset


class EEGNet(nn.Module):
    def __init__(self,i=2):
        super(EEGNet, self).__init__()

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=True),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        
        self.depthwiseConv1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=True),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.depthwiseConv2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 4), stride = (1, 4), padding=0),
            nn.Dropout(p=0.25)
        )

        self.separableConv1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=True),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
            
        self.separableConv2= nn.Sequential(
            nn.AvgPool2d(kernel_size=(1, 8), stride = (1, 8), padding=0),
            nn.Dropout(p=0.25)
        )

        self.classifly = nn.Sequential(
            nn.Linear(in_features=1472, out_features=2, bias=True)
        )

        self.nonlinearity = None

        if i == "ReLU":
            self.nonlinearity = nn.ReLU()
        elif i == "ELU":
            self.nonlinearity = nn.ELU(alpha=0.5)
        else:
            self.nonlinearity = nn.LeakyReLU()
    
    def forward(self,input):
        output = self.firstconv(input)
        output = self.depthwiseConv1(output)
        output = self.nonlinearity(output)
        output = self.depthwiseConv2(output)
        output = self.separableConv1(output)
        output = self.nonlinearity(output)
        output = self.separableConv2(output)
        output = output.view(output.size(0),-1)
        return self.classifly(output)

class DeepConvNet(nn.Module):
    def __init__(self,i=2):
        super(DeepConvNet, self).__init__()

        self.deepConvNet1 = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 60), bias=True),
            nn.Conv2d(25, 25, kernel_size=(2, 1)),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        
        self.deepConvNet2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0),
            nn.Dropout(p=0.5),
            nn.Conv2d(25, 50, kernel_size=(1, 50), bias=True),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.deepConvNet3  = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0),
            nn.Dropout(p=0.5),
            nn.Conv2d(50, 100, kernel_size=(1, 40), bias=True),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.deepConvNet4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0),
            nn.Dropout(p=0.5),
            nn.Conv2d(100, 200, kernel_size=(1, 30), bias=True),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
            
        self.deepConvNet5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0),
            nn.Dropout(p=0.5),
        )

        self.classifly = nn.Sequential(
            nn.Linear(in_features=2400, out_features=2, bias=True),
        )


        self.nonlinearity = None

        if i == "ReLU":
            self.nonlinearity = nn.ReLU()
        elif i == "ELU":
            self.nonlinearity = nn.ELU(alpha=1.0)
        else:
            self.nonlinearity = nn.LeakyReLU()
    
    def forward(self,input):
        output = self.deepConvNet1(input)
        output = self.nonlinearity(output)
        output = self.deepConvNet2(output)
        output = self.nonlinearity(output)
        output = self.deepConvNet3(output)
        output = self.nonlinearity(output)
        output = self.deepConvNet4(output)
        output = self.nonlinearity(output)
        output = self.deepConvNet5(output)

        output = output.view(output.size(0),-1)
        
        return self.classifly(output)

class MyDataset(Dataset):
    def __init__(self,data,label):
        self.data = torch.from_numpy(data).float()
        self.label = torch.from_numpy(label).float()
    def __getitem__(self,index):
        return self.data[index],self.label[index]
    def __len__(self):
        return len(self.data)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 128
leanring_rate = 1e-4
epochs = 1000

train_data, train_label, test_data, test_label = dataloader.read_bci_data()

my_train_data = MyDataset(train_data,train_label)

my_test_data = MyDataset(test_data,test_label)


train_data_loader = torch.utils.data.DataLoader(my_train_data,batch_size=batch_size,shuffle=True)

test_data_loader = torch.utils.data.DataLoader(my_test_data,batch_size=batch_size,shuffle=True)


x_axis = [i+1 for i in range(epochs)] 

net_name = ["DeepConvNet","EEGNet"]
nn_name = ["ReLU", "ELU", "LeakyReLU"]

def detailOfTensor(Tensor):
    print(Tensor,Tensor.dtype,Tensor.size(),Tensor.device)
    
def train():
    print("============Train============")
    for net_n in net_name:
        plt.title('Activation function comparision({})'.format(net_n),fontsize=25)
        plt.ylabel('Accuracy(%)',fontsize=16)
        plt.xlabel('Epoch',fontsize=16)
        num_line = 0
    
        for n in nn_name:
            best = 0
            if net_n == "DeepConvNet":
                net = DeepConvNet(n).to(device)
            else:
                net = EEGNet(n).to(device)
            print(net)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(net.parameters(), lr=leanring_rate)

            train_acc = []
            test_acc = []
            
            for i in range(1,epochs+1):
                train_sum = 0
                train_correct = 0
                test_sum = 0
                test_correct = 0
            
                for j, (data, target) in enumerate(train_data_loader):
                    data, target = data.to(device), target.to(device)
                    net.zero_grad()
                    output = net(data)

                    target = target.long()
                   
                    err = criterion(output, target)
                    for k in range(len(output)):
                        if output[k].argmax() == target[k]:
                            train_correct+=1
                        train_sum+=1
                    
                    err.backward()
                    optimizer.step()
                    optimizer.zero_grad()
            

                for j, (data, target) in enumerate(test_data_loader):
                    data, target = data.to(device), target.to(device)
                   
                    output = net(data)

                    target = target.long()
                    
                    for k in range(len(output)):
                        if output[k].argmax() == target[k]:
                            test_correct+=1
                        test_sum+=1
                
                train_acc.append(train_correct * 100.0 / train_sum)
                test_acc.append(test_correct * 100.0 / test_sum)
                if best < test_acc[-1]:
                    best = test_acc[-1]
                    torch.save(net.state_dict(), "{}_{}.h5".format(net_n,n) )

                if i % 100 == 0:
                    print("{} / {}, Train Accurancy : {:.4f}, Test Accurancy : {:.4f}.".format(i,epochs,train_acc[-1],test_acc[-1]))
            plt.plot(x_axis,train_acc, label= n + "_train", color='C' + str(num_line), linewidth=0.3)
            plt.plot(x_axis,test_acc, label= n + "_test", color='C' + str(num_line+1), linewidth=0.3)

            print("{} using {}, Best acc is {:.4f} in epoch {}".format(net_n,n,np.max(test_acc),np.argmax(test_acc)+1))       
            plt.plot(np.argmax(test_acc)+1, np.max(test_acc), marker='o', color= 'C' + str(num_line+1))
            num_line += 2 
        plt.legend(loc='lower right')
        plt.savefig(net_n + '.png',dpi = 100)
        plt.clf()

def test():
    print("============Test============")
    for net_n in net_name:
        for n in nn_name:
            test_sum = 0
            test_correct = 0 
            for j, (data, target) in enumerate(test_data_loader):
                if net_n == "DeepConvNet":
                    net = DeepConvNet(n).to(device)
                else:
                    net = EEGNet(n).to(device)

                data, target = data.to(device), target.to(device)
                net.load_state_dict(torch.load("{}_{}.h5".format(net_n,n)))
                net.eval()
                output = net(data)

                target = target.long()
                
                for k in range(len(output)):
                    if output[k].argmax() == target[k]:
                        test_correct+=1
                    test_sum+=1
            print("{} using {}, Acc is {}".format(net_n,n,test_correct * 100.0 / test_sum))
                

            
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-m", dest="mode" ,type=int)
   
    args = parser.parse_args()

    if args.mode == 0:
        train()
    else:
        test()
  
   
