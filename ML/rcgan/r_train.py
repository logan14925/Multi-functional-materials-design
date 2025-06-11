import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
import utils
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
file_folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_folder_path)
from modles import *

import time
plt.rcParams.update({
    'font.weight': 'bold',            # 加粗字体
    'axes.labelweight': 'bold',       # 加粗坐标轴标签
    'axes.titlesize': 26,             # 增加标题的字体大小
    'axes.labelsize': 18,             # 增加坐标轴标签字体大小
    'xtick.labelsize': 12,            # 增加X轴刻度字体大小
    'ytick.labelsize': 12,            # 增加Y轴刻度字体大小
    'legend.fontsize': 18,            # 增加图例字体大小
    'axes.titleweight': 'bold'        # 标题加粗
})

noisy_dim = config["rcgan_cellular"]["noise_size"]
lr_g = config["rcgan_cellular"]["lr_g"]
lr_d = config["rcgan_cellular"]["lr_d"]
lr_r = config["rcgan_cellular"]["lr_r"]
epochs = config["rcgan_cellular"]["epochs"]
z_dim = config["rcgan_cellular"]["latent_size"]
d_dim = config["rcgan_cellular"]["d_dim"]
# Optimizers
generator = Generator(noisy_dim, cellular_gan_output_dim, cellular_gan_input_dim).to(device)
discriminator = Discriminator(cellular_gan_input_dim, cellular_gan_output_dim).to(device)
regressor = Regressor(cellular_gan_input_dim, cellular_gan_output_dim).to(device)

optimizer_g = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))
optimizer_r = optim.Adam(regressor.parameters(), lr=lr_r)

scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_g, 'min',factor=0.2, patience=3, verbose=True, min_lr=1e-20)
scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_d, 'min',factor=0.2, patience=3, verbose=True, min_lr=1e-20)
scheduler_r = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_r, 'min',factor=0.2, patience=3, verbose=True, min_lr=1e-20)

# 初始化用于跟踪最小损失的变量
best_g_loss = float('inf')
best_d_loss = float('inf')
best_r_loss = float('inf')

# 用于保存所有的损失值
D_Losses, G_Losses, R_Losses = [], [], []
D_real_accuracies, D_fake_accuracies = [], []
start_time = time.time()
for epoch in range(epochs):
    # 每个epoch初始化损失
    D_Losses_ = []
    G_Losses_ = []
    R_Losses_ = []
    D_real_accuracies_ = []
    D_fake_accuracies_ = []
    
    # 开始训练
    generator.train()
    discriminator.train()
    regressor.train()

    with tqdm(enumerate(cellular_gan_train), total=len(cellular_gan_train), desc=f"Epoch {epoch+1}/{epochs}", ncols=100) as pbar:
        for batch_idx, (inputs, cond) in pbar:
            # 组织数据
            inputs = inputs.to(device)
            cond = cond.to(device)
            cond = cond.to(torch.float32)
            inputs = inputs.to(torch.float32)
            
            batch_size = inputs.size(0)
            batch_y = cond
            batch_z = torch.randn(batch_size, noisy_dim).to(device)
            d = batch_y[:, 0:1]
            # ---- Train Generator ----
            optimizer_g.zero_grad()

            # 生成假数据
            fake_data = generator(batch_z, batch_y)
            latent_fake, fake_pred = regressor(fake_data)
            latent_real, real_pred = regressor(inputs)

            # 计算生成器损失
            G_loss = F.mse_loss(fake_pred, batch_y) + F.mse_loss(fake_data, inputs)
            G_loss.backward(retain_graph=True)
            optimizer_g.step()

            G_Losses_.append(G_loss.item())

            # ---- Train Discriminator ----
            optimizer_d.zero_grad()

            real_data_pred = discriminator(latent_real, batch_y)
            fake_data_pred = discriminator(latent_fake.detach(), batch_y)

            D_real_loss = F.mse_loss(real_data_pred, torch.ones_like(real_data_pred) * 0.9)
            D_fake_loss = F.mse_loss(fake_data_pred, torch.zeros_like(fake_data_pred))
            D_loss = (D_real_loss + D_fake_loss) / 2
            D_loss.backward(retain_graph=True)
            optimizer_d.step()

            D_Losses_.append(D_loss.item())

            real_accuracy = ((real_data_pred > 0.5).float() == torch.ones_like(real_data_pred)).float().mean().item()
            fake_accuracy = ((fake_data_pred < 0.5).float() == torch.zeros_like(fake_data_pred)).float().mean().item()

            D_real_accuracies_.append(real_accuracy)
            D_fake_accuracies_.append(fake_accuracy)


            # ---- Train Regressor ----
            optimizer_r.zero_grad()

            R_loss = F.mse_loss(real_pred, batch_y)
            R_loss.backward()
            optimizer_r.step()

            R_Losses_.append(R_loss.item())

            # 更新进度条
            pbar.set_postfix(
                D_Loss=np.mean(D_Losses_),
                G_Loss=np.mean(G_Losses_),
                R_Loss=np.mean(R_Losses_)
            )

    # 记录每个epoch的平均损失
    avg_D_loss = np.mean(D_Losses_)
    avg_G_loss = np.mean(G_Losses_)
    avg_R_loss = np.mean(R_Losses_)

    avg_real_accuracy = np.mean(D_real_accuracies_)
    avg_fake_accuracy = np.mean(D_fake_accuracies_)
    
    D_real_accuracies.append(avg_real_accuracy)
    D_fake_accuracies.append(avg_fake_accuracy)
    
    D_Losses.append(avg_D_loss)
    G_Losses.append(avg_G_loss)
    R_Losses.append(avg_R_loss)

    # 如果当前生成器的损失较小，则保存生成器模型
    if avg_G_loss < best_g_loss:
        best_g_loss = avg_G_loss
        # 保存生成器
        torch.save(generator.state_dict(), best_g_model_path)
        print(f"Best Generator model saved with loss: {avg_G_loss:.4f}")

    # 如果当前判别器的损失较小，则保存判别器模型
    if avg_D_loss < best_d_loss:
        best_d_loss = avg_D_loss
        # 保存判别器
        torch.save(discriminator.state_dict(), best_d_model_path)
        print(f"Best Discriminator model saved with loss: {avg_D_loss:.4f}")

    # 如果当前回归器的损失较小，则保存回归器模型
    if avg_R_loss < best_r_loss:
        best_r_loss = avg_R_loss
        # 保存回归器
        torch.save(regressor.state_dict(), best_r_model_path)
        print(f"Best Regressor model saved with loss: {avg_R_loss:.4f}")
    
    # 打印损失信息
    print(f"Epoch {epoch+1}/{epochs} | "
          f"G Loss: {avg_G_loss:.4f} | "
          f"D Loss: {avg_D_loss:.4f} | "
          f"R Loss: {avg_R_loss:.4f} | "
          f"D Real Accuracy: {avg_real_accuracy:.4f} | "
          f"D Fake Accuracy: {avg_fake_accuracy:.4f}")

    # 更新学习率调度器
    scheduler_g.step(avg_G_loss)  # 这里使用生成器损失来更新学习率
    scheduler_d.step(avg_D_loss)  # 这里使用判别器损失来更新学习率
    scheduler_r.step(avg_R_loss)  # 这里使用回归器损失来更新学习率

# 训练完成
print('Training complete!')
# 记录训练结束时间
end_time = time.time()
times = end_time - start_time
print(times)
data = {
    'Epoch': list(range(1, epochs + 1)),
    'D_Loss': D_Losses,
    'G_Loss': G_Losses,
    'R_Loss': R_Losses,
    'D_Real_Accuracy': D_real_accuracies,
    'D_Fake_Accuracy': D_fake_accuracies
}

df = pd.DataFrame(data)

csv_file_path = "ML/rcgan/training_metrics.csv"
df.to_csv(csv_file_path, index=False, mode='w')  # mode='w'是默认的，会覆盖文件

print(f"Training metrics saved to {csv_file_path}")

# 在训练完成后统一绘制损失曲线
plt.figure(figsize=(30, 8))

# 第一个子图：D Loss
plt.subplot(1, 3, 1)  # 1行3列的第1个子图
plt.plot(D_Losses, label="D Loss", color='red')
plt.title("Discriminator Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# 第二个子图：G Loss
plt.subplot(1, 3, 2)  # 1行3列的第2个子图
plt.plot(G_Losses, label="G Loss", color='blue')
plt.title("Generator Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# 第三个子图：R Loss
plt.subplot(1, 3, 3)  # 1行3列的第3个子图
plt.plot(R_Losses, label="R Loss", color='green')
plt.title("Regressor Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()

# 新建一个新的 figure 用于绘制准确率曲线
plt.figure(figsize=(10, 8))

# 2. Discriminator Accuracy
plt.plot(D_real_accuracies, label="Real Accuracy", color='blue')
plt.plot(D_fake_accuracies, label="Fake Accuracy", color='red')
plt.title("Discriminator Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()

plt.show()

print("Final models have been saved.")
