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

# 绘图函数（已更新）
def plot_recon_and_pred(labels, recons, preds, save_path=None):
    if len(recons) == 0 or len(preds) == 0:
        print("Error: Empty input arrays")
        return

    # 计算输入与输出维度
    recon_array = recons[0].cpu().numpy() if hasattr(recons[0], 'cpu') else recons[0]
    input_dim = recon_array.shape[1] if len(recon_array.shape) > 1 else 1
    pred_array = preds[0]
    pred_dim = pred_array.shape[1] if len(pred_array.shape) > 1 else 1
    total_dims = input_dim + pred_dim

    fig, axes = plt.subplots(len(labels), total_dims, figsize=(18, 0.35 * len(labels) * total_dims))

    if len(labels) == 1:
        axes = axes.reshape(1, -1)

    for idx, (label, recon, pred) in enumerate(zip(labels, recons, preds)):
        recon_array = recon.cpu().numpy() if hasattr(recon, 'cpu') else recon
        pred_array = pred.cpu().numpy() if hasattr(pred, 'cpu') else pred

        # 绘制输入维度分布
        for i in range(input_dim):
            ax = axes[idx, i]
            ax.hist(recon_array[:, i], bins=25, color='blue', alpha=0.7, density=True)
            sns.kdeplot(recon_array[:, i], color='blue', ax=ax, linewidth=2)
            ax.set_title(f'Input {i}', fontsize=10)
            ax.set(xlabel=None, ylabel=None)

        # 绘制输出维度分布
        for j in range(pred_dim):
            ax_pred = axes[idx, input_dim + j]
            ax_pred.hist(pred_array[:, j], bins=30, color='green', alpha=0.7, density=True)
            sns.kdeplot(pred_array[:, j], color='green', ax=ax_pred, linewidth=2)
            ax_pred.axvline(x=label[j], color='red', linestyle='--', label=f'Label {j}: {label[j]:.2f}')
            ax_pred.set_title(f'Output {j}', fontsize=10)
            ax_pred.set(xlabel=None, ylabel=None)
            ax_pred.legend(loc='upper right', fontsize=8)

    for ax_row in axes:
        for ax in (ax_row if isinstance(ax_row, np.ndarray) else [ax_row]):
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

# 读取配置
noisy_dim = config["rcgan_cellular"]["noise_size"]
epochs = config["rcgan_cellular"]["epochs"]
z_dim = config["rcgan_cellular"]["latent_size"]
sample_num = config["rcgan_cellular"]["sample_num"]
label_propertys = [[0.5, 0.4], [0.5, 0.1], [0.7, 0.3], [0.3, 0.7]]
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

# 数据生成与预测
recons, preds, conds = [], [], []
start_time = time.time()

for label_property in label_propertys:
    # 正确构造 cond 张量
    cond = torch.tensor(label_property, dtype=torch.float32, device=device).unsqueeze(0).repeat(sample_num, 1)
    z = torch.randn(sample_num, noisy_dim).to(device)

    with torch.no_grad():
        generated_data = generator(z, cond)
        prediction = forward_net(generated_data)

    pred = prediction.cpu().numpy()

    recons.append(generated_data)
    preds.append(pred)
    conds.append(label_property)


end_time = time.time()
print(f"Total time taken to generate all label corresponding data: {end_time - start_time:.4f} seconds")

# 绘制图像
plot_recon_and_pred(label_propertys, recons, preds, None)
