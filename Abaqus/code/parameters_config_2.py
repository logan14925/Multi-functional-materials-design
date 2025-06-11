import numpy as np
import pandas as pd
import json
import math

# 设置随机种子以确保结果可重复
np.random.seed(14)  # 你可以选择任何整数作为种子值

# 读取 JSON 文件中的参数
config_path = 'Abaqus/FEM_config.json'
with open(config_path, 'r') as file:
    config_params = json.load(file)

result = np.empty((0, 9))  # 增加一列用于index
max_attempts = 1000000  # 防止无限循环
attempt_count = 0

max_radius = config_params["max_radius"]
max_length = config_params["max_length"]
x_array = config_params["x_array"]
y_array = config_params["y_array"]

while len(result) < 500 and attempt_count < max_attempts:
    # 逐参数生成（保持原有范围）
    thickness_values = [0.4, 0.8, 1.2] + list(np.arange(1.3, 2.1, 0.1))  # 合并多个步长的数值
    thickness = np.random.choice(thickness_values)
    radius = np.random.uniform( thickness, 4 * thickness)
    length_x = np.random.uniform(2 * radius + thickness, 6 * radius)
    length_y = np.random.uniform(2 * radius + thickness, 6 * radius)

    arc_y = np.random.uniform(-50, length_y/2)
    arc_x = np.random.uniform(3*length_x/4, 50)

    arc_y_r = ((length_x / 2)**2 + (length_y / 2 - arc_y)**2)**0.5 + radius
    arc_y_theta = math.atan(length_x / (length_y - 2 * arc_y))
    arc_x_r = ((length_y / 2)**2 + (length_x / 2 - arc_x)**2)**0.5 + radius
    arc_x_theta = math.atan(length_y / (2 * arc_x - length_x))
    
    arc_length = 8 * arc_x_theta*arc_x_r + 8 * arc_x_theta *arc_x_r
    circle_length = 4 * 2*math.pi*radius
    equal_S = (arc_length + circle_length) * thickness * x_array * y_array *0.855
    equal_length = 16 * length_y
    # 曲率计算
    k_x = 1 / arc_x_r
    k_y = 1 / arc_y_r
    
    # 条件判断（保持原有逻辑）
    condition1 = length_y > arc_y_r + arc_y + thickness
    condition2 = arc_x_r + thickness<arc_x
    #保证不会出现干涉的问题
    condition3 = k_x >= 0.05
    condition4 = k_y >= 0.05
    condition5 = radius <= max_radius
    condition6 = length_x <= max_length
    condition7 = length_y <= max_length
    # 若满足条件则保存
    if condition1 and condition2 and condition3 and condition4 and condition5 and condition6 and condition7:
        index = len(result)  # 获取当前行index
        new_row = np.array([[index, thickness, radius, length_x/2, length_y/2, arc_x, arc_y, equal_S, equal_length]])
        new_row = np.round(new_row, 4)  # 保留3位有效数字
        result = np.vstack((result, new_row))
    
    attempt_count += 1

# 处理未生成足够数据的情况
if len(result) < 500:
    print(f"警告：仅生成 {len(result)} 条数据（尝试次数：{attempt_count}）")
else:
    print("成功生成 500 条数据")

# 保存结果（保留原有方式）
df = pd.DataFrame(result, columns=['index', 'thickness', 'radius', 'length_x/2', 'length_y/2', 'arc_x', 'arc_y',  'equal_S', 'equal_length'])
df.to_csv('Abaqus/paras2.csv', index=False)