import numpy as np
import pandas as pd
import json
import math

# 设置随机种子以确保结果可重复
np.random.seed(13)  # 你可以选择任何整数作为种子值

# 读取 JSON 文件中的参数
config_path = 'Abaqus/FEM_config.json'
with open(config_path, 'r') as file:
    config_params = json.load(file)

result = np.empty((0, 4))
max_attempts = 10000  # 防止无限循环
attempt_count = 0

max_radius = config_params["max_radius"]
max_length = config_params["max_length"]
x_array = config_params["max_length"]
while len(result) < 10 and attempt_count < max_attempts:
    # 逐参数生成（保持原有范围）
    thickness = np.random.randint(1, 6) * 0.4
    radius = np.random.uniform(2 * thickness, 4 * thickness)
    length = np.random.uniform(2 * radius + 2, 6 * radius)
    arc = np.random.uniform(-50, length/2)
    
    arc_r = ((length / 2)**2 + (length / 2 - arc)**2)**0.5 + radius
    arc_theta = math.atan(length / (length - 2 * arc))
    arc_length = 8 * arc_theta / math.pi * 2*math.pi*arc_r
    circle_length = 4 * 2*math.pi*radius
    equal_S = (arc_length + circle_length) * thickness * 50
    equal_length = 20 * length
    # 曲率计算
    k = 1 / np.sqrt(arc**2 + (length**2)/4)
    
    # 条件判断（保持原有逻辑）
    condition1 = (7 * length / 8) > (np.sqrt(length**2/4 + (length/2 - arc)**2) + radius + arc)
    #保证不会出现干涉的问题
    condition2 = k >= 0.05
    condition3 = radius <= max_radius
    condition4 = length <= max_length
    
    # 若满足条件则保存
    if condition1 and condition2 and condition3 and condition4:
        new_row = np.array([[thickness, radius, length, arc, equal_S, equal_length]])
        result = np.vstack((result, new_row))
    
    attempt_count += 1

# 处理未生成足够数据的情况
if len(result) < 500:
    print(f"警告：仅生成 {len(result)} 条数据（尝试次数：{attempt_count}）")
else:
    print("成功生成 500 条数据")

# 保存结果（保留原有方式）
df = pd.DataFrame(result, columns=['thickness', 'radius', 'length', 'arc', 'equal_S', 'equal_length'])
df.to_csv('paras.csv', index=False)