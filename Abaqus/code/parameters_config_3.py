import numpy as np
import pandas as pd
import json
import math
from scipy.optimize import fsolve

# 设置随机种子以确保结果可重复
np.random.seed(14)  # 你可以选择任何整数作为种子值

# 读取 JSON 文件中的参数
config_path = 'Abaqus/FEM_config.json'
with open(config_path, 'r') as file:
    config_params = json.load(file)

result = np.empty((0, 10))  # 增加一列用于index
max_attempts = 1000000  # 防止无限循环
attempt_count = 0

num_ = 512

max_radius = config_params["max_radius"]
max_length = config_params["max_length"]
x_array = config_params["x_array"]
y_array = config_params["y_array"]


while len(result) < num_ and attempt_count < max_attempts:
    # 逐参数生成（保持原有范围）
    thickness_values = [0.4, 0.8, 1.2] + list(np.arange(1.3, 2.1, 0.1))  # 合并多个步长的数值
    thickness = np.random.choice(thickness_values)
    radius = np.random.uniform( thickness, 10 * thickness)
    length_x = np.random.uniform(2 * radius + thickness, 6 * radius)/2
    length_y = np.random.uniform(2 * radius + thickness, 6 * radius)/2

    # # 定义两个区间
    # interval1 = (thickness, length_x - radius - thickness)
    # interval2 = (length_x - radius + thickness, length_x)

    # # 随机选择一个区间，再生成区间内的随机数
    # d1x = np.random.choice([
    #     np.random.uniform(*interval1),
    #     np.random.uniform(*interval2)
    # ])
    
    d1x = np.random.uniform(thickness, length_x-radius-thickness)
    def equations(vars):
        d2x, d2y = vars
        eq1 = (d2y / (d2x - d1x)) - ((length_x - d2x) / (d2y - length_y))
        eq2 = (d2y - length_y)**2 + (d2x - length_x)**2 - radius**2
        return [eq1, eq2]
    initial_guess = [0, 0]  
    solution = fsolve(equations, initial_guess)
    d2x_sol, d2y_sol = solution
    
    arc_length = (2*math.pi*radius + 2*length_x + 2*(length_y**2+(length_x-d1x)**2-radius**2)**0.5)*4
    equal_S = arc_length * thickness * x_array * y_array *0.855
    equal_length = 4*x_array * length_x

    #保证不会出现干涉的问题
    condition1 = radius <= max_radius
    condition2 = 2*length_x <= max_length
    condition3 = 2*length_y <= max_length
    condiiton4 = d2x_sol<length_x
    condition5 = length_x - radius - thickness > thickness
    condition6 = length_x > length_x - radius + thickness
    condition7 = abs((d2y_sol - length_y)**2 + (d2x_sol - length_x)**2 - radius**2) < 1e-3
    condition8 = abs((d2y_sol / (d2x_sol - d1x)) - ((length_x - d2x_sol) / (d2y_sol - length_y))) < 1e-3
    if condition1 and condition2 and condition3 and condiiton4 and condition5 and condition6 and condition7 and condition8:
        index = len(result)  # 获取当前行index
        new_row = np.array([[index, thickness, radius, length_x, length_y, d1x, equal_S, equal_length, d2x_sol, d2y_sol]])
        new_row = np.round(new_row, 4)  # 保留3位有效数字
        result = np.vstack((result, new_row))
    
    attempt_count += 1

# 处理未生成足够数据的情况
if len(result) < num_:
    print(f"警告：仅生成 {len(result)} 条数据（尝试次数：{attempt_count}）")
else:
    print("成功生成 {} 条数据".format(num_))

# 保存结果（保留原有方式）
df = pd.DataFrame(result, columns=['index', 'thickness', 'radius', 'length_x/2', 'length_y/2', 'd',  'equal_S', 'equal_length', 'd2x', 'd2y'])
df.to_csv('Abaqus/paras3.csv', index=False)