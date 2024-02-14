import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from datetime import datetime
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import sys
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
#print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(14, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        
        logits = self.linear_relu_stack(x)
        return logits

class dataset(Dataset):
    def __init__(self, is_train = False):
        df = pd.read_csv("data.csv")
        
        df["winner_shot_type"] = df["winner_shot_type"].replace('F','1')
        df["winner_shot_type"] = df["winner_shot_type"].replace('B','1')
        
        df_bad = df[df["speed_mph"].isnull()]
        df = df[df["speed_mph"].isnull() == False]
        if is_train:
            df = df.iloc[0:6000,:]
        else:
            df = df.iloc[6001:,:]
        
        self.data = df.iloc[:,[4,5,6,16,17,20,21,22,23,24,29,30,39,40]].values.astype(np.float32)
        self.label = np.column_stack((df["speed_mph"].values, df["rally_count"].values))
        #print(self.label.shape)
        #sys.exit()
		
   
    def __getitem__(self,index):
        label = self.label[index]
        data = self.data[index]
        return data, label

    def __len__(self):
        return len(self.data)




def train():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    end_epoch = 20
    l1loss = torch.nn.L1Loss()
    train_loss_map = []
    valid_loss_map = []
    with torch.device("cpu"):
        ## model
        model = NeuralNetwork() # 定义模型
        
        # optimizer & lr_scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

        #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.8) # 定义学习率
        # lr_scheduler = lr_decay()  # 也可以是自己定义的学习率下降方式,比如定义了一个列表

        #if resume:  # restore from checkpoint
           # model, optimizer = restore_from(model, optimizer, ckpt) # 恢复训练状态

        # load train data
        trainset = dataset(is_train = True)
        trainloader = DataLoader(dataset=trainset, batch_size=4,
                         shuffle=False, num_workers=0,
                         drop_last=True, pin_memory=True)
        
        validset = dataset(is_train = False)
        validloader = DataLoader(dataset=validset, batch_size=4,
                         shuffle=False, num_workers=0,
                         drop_last=True, pin_memory=True)

        ### logs
        #logger = create_logger()  # 自己定义创建的log日志
        #summary_writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp)) # tensorboard


        ### start train
        for epoch in range(end_epoch):

            #lr_scheduler.step() # 更新optimizer的学习率，一般以epoch为单位，即多少个epoch后换一次学习率

            train_loss = 0
            model.train()

            ## train
            for i, data in enumerate(trainloader):
                inputs, target = data
                optimizer.zero_grad() #使用之前先清零
                output = model(inputs)
                loss = l1loss(output, target)  # 自己定义损失函数
                loss.sum().backward() # loss反传，计算模型中各tensor的梯度
                optimizer.step() #用在每一个mini batch中，只有用了optimizer.step()，模型才会更新
                train_loss += loss
            train_loss /= i # 对各个mini batch的loss求平均

            ## eval，不需要梯度反传
            valid_loss = 0
            model.eval()  # 注意model的模式从train()变成了eval()
            for i, data in enumerate(validloader):
                inputs, target = data
                optimizer.zero_grad()
                output = model(inputs)
                loss = l1loss(output, target)  # 自己定义损失函数
                valid_loss += loss
                
            valid_loss /= i
            
            train_loss_map.append(float(train_loss))
            valid_loss_map.append(float(valid_loss))
            print("Train:", float(train_loss),"Valid:", float(valid_loss))
            #summary_writer.add_scalars('loss', {'train_loss': train_loss, 'valid_loss': valid_loss}, epoch) #写入tensorboard

            if (epoch + 1) == end_epoch: # 保存模型
                model_path = 'model_{}_{}'.format(timestamp, epoch)
                torch.save(model,model_path)
                plt.plot(train_loss_map)
                plt.plot(valid_loss_map)
                plt.xlabel("epoch")
                plt.ylabel("loss")
                plt.legend(["train_loss","valid_loss"])
                
if True:             
    train()
    sys.exit()

df = pd.read_csv("data.csv")

df["winner_shot_type"] = df["winner_shot_type"].replace('F','1')
df["winner_shot_type"] = df["winner_shot_type"].replace('B','1')

df_bad = df[df["speed_mph"].isnull()]

inputs = df_bad.iloc[:,[4,5,6,16,17,20,21,22,23,24,29,30,39,40]].values.astype(np.float32)
#print(df.iloc[:,[4,5,6,16,17,20,21,22,23,24,29,30,39,40,41]])
model = torch.load('model_20240204_153233_19')
model = model.eval()

output = model(torch.from_numpy(inputs))
output_np = output.detach().cpu().numpy().round().astype(int)
df_bad.loc[:,["speed_mph", "rally_count"]] = output_np

df[df["speed_mph"].isnull()] = df_bad
print(df.iloc[46,0])

df.to_csv("data_fill.csv",index=False)