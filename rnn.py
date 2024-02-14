import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import sys
import matplotlib.pyplot as plt
import pickle
import os
from tkinter import messagebox

torch.manual_seed(1)

df = pd.read_csv("data_fill_main.csv")

df = df.fillna(0)

df = df.replace({"winner_shot_type": {"F": "1", "B": "2"},
                 "serve_depth": {"NCTL": "0", "CTL": "1"},
                 "serve_width": {"B": "0", "BC": "1", "BW": "2", "C": "3", "W": "4"},
                 "return_depth": {"ND": "1", "D": "2"}})
        
winlose = df.loc[:,["point_victor"]].drop(0)

t = 0
while t < df.shape[0]:
    pt0 = int(df.loc[t, ['p1_points_won']].iloc[0])
    pt1 = int(df.loc[t + 1, ['p1_points_won']].iloc[0]) if t + 1 < df.shape[0] else 0
    if pt1 < pt0:
        df = df.drop(t)
        if t + 1 < df.shape[0]:
            winlose = winlose.drop(t + 1)
    t += 1

winlose = winlose.values
winlose = (winlose == 1).astype(np.float32)
#inps = [4,5,6,13,14,15,20,21,22,23,24,27,28,29,30,39,40,41,42,43,44,45]
inps = [4,5,6,13,14,15,20,21,22,23,24,27,28,29,30,39,40,41,43,44,45]
x = df.iloc[:,inps].values.astype(np.float32)
x_var = np.var(x, axis=0)
x_mean = np.mean(x, axis=0)
x_norm = (x - x_mean) / x_var
x_size = x.shape[1]

device = "cuda" if torch.cuda.is_available() else "cpu"
#print(f"Using {device} device")

hidden_size = 32
num_layers = 2
lr = 1e-3
weight_decay = 5e-4
step_size = 100
gamma = 0.99
batch_size = 16

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.lstm = torch.nn.LSTM(x_size, hidden_size,num_layers=num_layers)
        self.linear = torch.nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        linear_out = self.linear(lstm_out)
        pred_out = nn.functional.sigmoid(linear_out)
        return pred_out

class dataset(Dataset):
    def __init__(self, is_train = False):
        if is_train:
            x_slice = x_norm[0:5500,:]
            winlose_slice = winlose[0:5500]
        else:
            x_slice = x_norm[5501:6500,:]
            winlose_slice = winlose[5501:6500]
        self.data = x_slice
        self.label = winlose_slice
		
   
    def __getitem__(self,index):
        label = self.label[index]
        data = self.data[index]
        return data, label

    def __len__(self):
        return len(self.data)



def train():
    timestamp = datetime.now().strftime('%m%d_%H%M%S')
    end_epoch = 100
    lnloss = torch.nn.L1Loss()
    train_loss_map = []
    valid_loss_map = []
    
    signature = timestamp +" hidden_size="+str(hidden_size) + " num_layers=" + str(num_layers)+ " lr="+str(lr)+ " weight_decay="+ str(weight_decay)+ " step_size="+str(step_size)+ " gamma="+ str(gamma)+ " batch_size="+ str(batch_size) + "\n"
    with open("./model_list.txt","a") as fi:
        fi.write(signature)
    print(signature)
    with torch.device("cpu"):
        start_epoch = 0
        ## model
        model = NeuralNetwork() # 定义模型
        
        # optimizer & lr_scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=step_size, gamma=gamma) # 定义学习率

        # load train data
        trainset = dataset(is_train = True)
        trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=False)
        
        validset = dataset(is_train = False)
        validloader = DataLoader(dataset=validset, batch_size=batch_size, shuffle=False)

        ### logs
        #logger = create_logger()  # 自己定义创建的log日志
        #summary_writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp)) # tensorboard
        
        if False:
            start_epoch = 0
            start_time = "0205_133310"
            timestamp = start_time
            path_checkpoint = "./chkpt/{}/checkpoint_{}.pkl".format(start_time,start_epoch)
            checkpoint = torch.load(path_checkpoint)#load文件
            model.load_state_dict(checkpoint['model_state_dict'])#恢复模型参数
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])#恢复优化器参数
            start_epoch = checkpoint['epoch']#设置要恢复的epoch
            lr_scheduler.last_epoch = start_epoch#设置学习率
            with open("./chkpt/{}/train_loss_{}.pkl".format(start_time,start_epoch), 'rb') as fi:
                train_loss_map=pickle.load(fi)
            with open("./chkpt/{}/valid_loss_{}.pkl".format(start_time,start_epoch), 'rb') as fi:
                valid_loss_map=pickle.load(fi)
            
        ### start train
        for epoch in range(start_epoch, end_epoch):
            
            #lr_scheduler.step() # 更新optimizer的学习率，一般以epoch为单位，即多少个epoch后换一次学习率

            train_loss = 0
            model.train()

            ## train
            for i, data in enumerate(trainloader):
                inputs, target = data
                #target = target.unsqueeze(1)
                optimizer.zero_grad() #使用之前先清零
                output = model(inputs)
                #print(output,target)
                #print(output.dtype, target.dtype)
                #sys.exit()
                loss = lnloss(output, target)  # 自己定义损失函数
                loss.sum().backward() # loss反传，计算模型中各tensor的梯度
                optimizer.step() #用在每一个mini batch中，只有用了optimizer.step()，模型才会更新
                train_loss += loss
            train_loss /= i # 对各个mini batch的loss求平均

            ## eval，不需要梯度反传
            valid_loss = 0
            model.eval()  # 注意model的模式从train()变成了eval()
            for i, data in enumerate(validloader):
                inputs, target = data
                #target = target.unsqueeze(1)
                optimizer.zero_grad()
                output = model(inputs)
                loss = lnloss(output, target)  # 自己定义损失函数
                valid_loss += loss
                
            valid_loss /= i
            
            
            lr_scheduler.step()
            train_loss_map.append(float(train_loss))
            valid_loss_map.append(float(valid_loss))
            
            
            if epoch % 10 == 9:
                print("epoch:", epoch, "Train:", float(train_loss),"Valid:", float(valid_loss))
                
            if epoch % 50 == 49:
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), 
                    "epoch": epoch}
                path_checkpoint = "./chkpt/{}/checkpoint_{}.pkl".format(timestamp,epoch)
                if not os.path.exists("./chkpt/{}".format(timestamp)):
                    os.makedirs("./chkpt/{}".format(timestamp))
                with open("./chkpt/{}/train_loss_{}.pkl".format(timestamp,epoch), 'wb') as fi:
                    pickle.dump(train_loss_map, fi)
                with open("./chkpt/{}/valid_loss_{}.pkl".format(timestamp,epoch), 'wb') as fi:
                    pickle.dump(valid_loss_map, fi)
                    
                torch.save(checkpoint, path_checkpoint)
                plt.plot(train_loss_map)
                plt.plot(valid_loss_map)
                plt.xlabel("epoch")
                plt.ylabel("loss")
                plt.legend(["train_loss","valid_loss"])
                plt.title(signature)
                plt.pause(0.01)
            
            #summary_writer.add_scalars('loss', {'train_loss': train_loss, 'valid_loss': valid_loss}, epoch) #写入tensorboard

            if epoch % 50 == 49: # 保存模型
                location = "speed"
                if not os.path.exists("./model/{}".format(location)):
                    os.makedirs("./model/{}".format(location))
                model_path = './model/{}/rnnmodel_{}'.format(location,epoch)
                torch.save(model,model_path)
                

train()

