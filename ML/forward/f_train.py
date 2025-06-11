import torch
import numpy as np
from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F
from torch.nn.functional import normalize
import os
import sys
file_folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_folder_path)
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import time
from torch.autograd import Variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib.pyplot as plt
from modles import *
from mpl_toolkits.mplot3d import Axes3D  

plt.rcParams.update({
    'font.weight': 'bold',            # 加粗字体
    'axes.labelweight': 'bold',       # 加粗坐标轴标签
    'axes.titlesize': 12,             # 增加标题的字体大小
    'axes.labelsize': 12,             # 增加坐标轴标签字体大小
    'xtick.labelsize': 12,            # 增加X轴刻度字体大小
    'ytick.labelsize': 12,            # 增加Y轴刻度字体大小
    'legend.fontsize': 12,            # 增加图例字体大小
    'axes.titleweight': 'bold',        # 标题加粗
})

forward_regression_net = forward_regression(forward_input_dim, forward_output_dim).to(device)

lr = config["cellular_forward"]["lr"]

optimizer = torch.optim.SGD(forward_regression_net.parameters(), lr=lr, weight_decay=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.2, patience=3, verbose=True, min_lr=1e-20)

loss_func = torch.nn.MSELoss()
num_epochs = 200
total_train_step = 0
train_losses = []
now_loss = []
test_loss_list = []
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
plt.ion()  # 开启交互模式
plt.show()
start_time = time.time()
for i in range(num_epochs):
    print("-------第{}轮训练开始-------".format(i+1))

    forward_regression_net.train() 
    #更改这里的数据集
    for data in forward_train:
        inputs, targets = data
        inputs = inputs.float().to(device)
        targets = targets.float().to(device)
        
        optimizer.zero_grad()
        prediction = forward_regression_net(inputs)
        loss = loss_func(prediction, targets)
        # print('prediction is {}\ntarget is {}'.format(prediction, targets))
        loss.backward()
        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数:{}, Loss:{}".format(total_train_step, loss.item()))
            # print('prediction is {}\ntarget is {}'.format(prediction, targets))
    train_losses.append(loss.item())
    print(train_losses[-1])
    if len(train_losses) >= 10:
        now_loss = train_losses[-5:]
        now_average_loss = sum(now_loss) / len(now_loss)
        scheduler.step(now_average_loss)
        
    val_losses = []
    forward_regression_net.eval() 

    with torch.no_grad():
        for data in forward_val:
            inputs, targets = data
            inputs = inputs.float().to(device)
            targets = targets.float().to(device)
            
            prediction = forward_regression_net(inputs)
            # if i == num_epochs -1:
            #     print('pre = {}'.format(prediction), 'target = {}'.format(targets), '\n')
            loss = loss_func(prediction, targets)
            val_losses.append(loss.item())

    average_test_loss = sum(val_losses) / len(val_losses)
    test_loss_list.append(average_test_loss)
    print("测试集损失: {}".format(average_test_loss))

end_time = time.time()
times = end_time - start_time
print(times)

plt.figure(figsize=(10, 8))
plt.plot(train_losses, label="Training Loss")
plt.plot(test_loss_list, label="Validation Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
# plt.title('Training and Test Loss Curve')
plt.legend()
plt.tight_layout()
plt.show()
train_losses = torch.tensor(train_losses)
train_losses = torch.unsqueeze(train_losses, dim=1)

best_loss, index = torch.min(train_losses, dim=0)

best_loss_epoch = index + 1

best_model = forward_regression(forward_input_dim, forward_output_dim).to(device)
best_model.load_state_dict(forward_regression_net.state_dict())
torch.save(best_model.state_dict(), forward_pth_path)

print(f"低损失的模型已保存为'MaterialNet_model_best.pth'，损失值为: {best_loss}，对应的epoch为: {best_loss_epoch}")



