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
import numpy as np
import pandas as pd
import seaborn as sns
import time
from sklearn.metrics import r2_score
from scipy.ndimage import gaussian_filter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 文件路径配置
file_folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_folder_path)
from modles import *

# 读取配置
noisy_dim = config["rcgan_cellular"]["noise_size"]
epochs = config["rcgan_cellular"]["epochs"]
z_dim = config["rcgan_cellular"]["latent_size"]
sample_num = config["rcgan_cellular"]["sample_num"]

label_d = config["rcgan_cellular"]["label_d"]
xlsx_base_save_path = config["rcgan_cellular"]["generated_data_path"]
error_threshold = config["rcgan_cellular"]["error_threshold"]
visualization_path = config["rcgan_cellular"]["visualization_path"]
label_d = torch.full((sample_num, 1), label_d).to(device)

# 设置绘图样式
plt.rcParams.update({
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titlesize': 26,
    'axes.labelsize': 18,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.titleweight': 'bold'
})

# 加载模型
generator = Generator(noisy_dim, cellular_gan_output_dim, cellular_gan_input_dim).to(device)
generator.load_state_dict(torch.load(best_g_model_path))
generator.eval()

forward_net = forward_regression(forward_input_dim, forward_output_dim).to(device)
forward_net.load_state_dict(torch.load(forward_pth_path))
forward_net.eval()

output = normalized_cellular_gan_output.cpu().numpy()
# === 开始绘图 ===
plt.figure(figsize=(8, 6))

# 绘制真实数据点（蓝色正方形）
plt.scatter(output[:, 0], output[:, 1], c='blue', marker='s', alpha = 0.5,  label='real data')

# 构建二维网格: 0~1 每隔0.2
grid_vals = np.linspace(0, 1, 50 + 1)
recons, preds, conds = [], [], []
start_time = time.time()

for moduli in grid_vals:
    for poisson in grid_vals:
        label = np.array([moduli, poisson])
        cond = torch.tensor(label, dtype=torch.float32, device=device).unsqueeze(0).repeat(sample_num, 1)
        z = torch.randn(sample_num, noisy_dim).to(device)

        with torch.no_grad():
            generated_data = generator(z, cond)
            prediction = forward_net(generated_data)

        pred = prediction.cpu().numpy()
        # 计算误差（逐个预测与label之间的绝对差）
        abs_errors = np.abs(pred - label)

        # 条件：两个维度误差均 < 0.1
        valid_mask = np.mean(abs_errors, axis=1) < 0.1

        if np.any(valid_mask):
            best_error = abs_errors[valid_mask][0].mean()
            color_intensity = best_error  # 0 越深红，1 越白

            plt.scatter(
                label[0], label[1],
                color=(1, color_intensity, color_intensity),
                marker='o',
                s=40,
                alpha=0.3,
                label='inverse_points' if (moduli == 0 and poisson == 0) else None
            )

# 图形修饰
plt.xlabel("Moduli")
plt.ylabel("Poisson Ratio")
plt.title("real_VS_inverse")

plt.legend()
plt.tight_layout()
plt.show()