import torch
import os
import sys
import time  # 导入 time 模块
file_folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_folder_path)
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib.pyplot as plt
from modles import *
import random
from sklearn.metrics import r2_score

seed = 55
torch.manual_seed(seed)
# 设置全局字体和大小，同时加粗标题和坐标轴
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


all_predictions = []
all_targets = []

# Initialize variable to track total time
total_time = 0

# Extract results from 30 batches
for _ in range(10):
    random_batch = random.choice(list(iter(forward_train)))
    x, targets = random_batch
    forward_net = forward_regression(forward_input_dim, forward_output_dim).to(device)
    forward_net.load_state_dict(torch.load(forward_pth_path))
    forward_net.eval()
    x = x.to(device)
    targets = targets.to(device)
    
    # Record start time
    start_time = time.time()
    
    # Use the model for prediction
    with torch.no_grad():
        prediction = forward_net(x)
    
    # Record end time and calculate the time taken for this batch
    end_time = time.time()
    total_time += (end_time - start_time)  # Accumulate the total time
    print('total time is ', total_time)
    # Denormalize predictions and targets
    # prediction = d_minmax_normal(prediction, min_mecha, max_mecha)
    # targets = d_minmax_normal(targets, min_mecha, max_mecha)
    
    # Append predictions and targets to the lists
    all_predictions.append(prediction.cpu().numpy())
    all_targets.append(targets.cpu().numpy())

# 将预测和目标拼接成完整数组
all_predictions = np.concatenate(all_predictions, axis=0)  # shape: (N, D)
all_targets = np.concatenate(all_targets, axis=0)          # shape: (N, D)

# 还原预测和目标数据的实际数值（反归一化）
real_preds = d_minmax_normal(torch.tensor(all_predictions, device=device), min_forward_output, max_forward_output)
real_targets = d_minmax_normal(torch.tensor(all_targets, device=device), min_forward_output, max_forward_output)

# 转为 NumPy 数组
real_preds_np = real_preds.cpu().numpy()
real_targets_np = real_targets.cpu().numpy()

# 获取输出维度
output_dim = real_preds_np.shape[1]

# 设置图像
fig, axes = plt.subplots(nrows=int(np.ceil(output_dim / 2)), ncols=2, figsize=(12, 5 * int(np.ceil(output_dim / 2))))
axes = axes.flatten()

# 逐维绘图和计算 R²
for i in range(output_dim):
    ax = axes[i]
    y_true = real_targets_np[:, i]
    y_pred = real_preds_np[:, i]
    
    # 计算 R²
    r2 = r2_score(y_true, y_pred)
    
    # 动态设置坐标范围
    data_min = min(np.min(y_true), np.min(y_pred))
    data_max = max(np.max(y_true), np.max(y_pred))
    data_range = data_max - data_min
    x_start = data_min - 0.1 * data_range
    x_end = data_max + 0.1 * data_range
    
    ax.scatter(y_true, y_pred, color='red', alpha=0.7, label='Prediction')
    ax.plot([x_start, x_end], [x_start, x_end], color='black', linestyle='--', label='y = x')
    ax.set_xlim(x_start, x_end)
    ax.set_ylim(x_start, x_end)
    ax.set_xlabel('Target')
    ax.set_ylabel('Prediction')
    ax.set_title(f'Output {i+1} | R²: {r2:.4f}')
    ax.legend()

# 如果子图数大于实际维度，关闭多余子图
for j in range(output_dim, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Print total time taken for forward passes
print(f"Total time taken for forward passes: {total_time:.4f} seconds")
