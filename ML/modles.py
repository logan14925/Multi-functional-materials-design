import torch
import numpy as np
from torch.utils.data import Dataset, random_split
from torch import nn
import torch.nn.functional as F
import os
import sys
file_folder_path = os.path.dirname(os.path.abspath(__file__))
print(file_folder_path)
sys.path.append(file_folder_path)
from torch.utils.data import DataLoader
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib.pyplot as plt
import json
import pandas as pd


torch.manual_seed(43)
class MyDataset(Dataset):
    """Coder在写这段代码的时候满脑都是小恐龙
    
    Args:
        Dataset (class): torch
    """
    def __init__(self, x, y):
        """_summary_

        Args:
            x (numpy): 输入层数据，未经归一化！！！
            y (numpy): 输出层数据，未经归一化！！！
        """
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        """X 进行了归一化， y没有进行归一化
        """
        return self.x[index], self.y[index]

    # def get_MINMAX_normal_data(self):
    #     """_summary_
    #     """
    #     self.normal_type = "min-max"
    #     self.get_min_max_data()
    #     self.normal_x = torch.div((self.x - self.min_x), (self.max_x - self.min_x))
    #     self.normal_y = torch.div((self.y - self.min_y), (self.max_y - self.min_y))

    # def z_score_normal(self):
    #     self.normal_type = "z-score"
    #     self.mean_x = torch.mean(self.x, dim=0)[0]
    #     self.std_x = torch.std(self.x, dim=0)
    #     self.mean_y = torch.mean(self.y, dim=0)
    #     self.std_y = torch.std(self.y, dim=0)

    #     self.normal_x = (self.x - self.mean_x) / self.std_x
    #     self.normal_y = (self.y - self.mean_y) / self.std_y


    # def get_min_max_data(self):
    #     """MIN-MAX归一化, 计算min_x, max_x, min_y, max_y
    #     """
    #     self.min_x = torch.min(self.x, dim=0)[0]
    #     self.max_x = torch.max(self.x, dim=0)[0]
    #     self.min_y = torch.min(self.y, dim=0)[0]
    #     self.max_y = torch.max(self.y, dim=0)[0]

    # def get_mean_std_data(self):
    #     """Z-Score归一化,计算mean_x, std_x, mean_y, std_y
    #     """
    #     self.mean_x = torch.mean(self.x, dim=0)[0]
    #     self.std_x = torch.std(self.x, dim=0)
    #     self.mean_y = torch.mean(self.y, dim=0)
    #     self.std_y = torch.std(self.y, dim=0)

def z_score_normal(x):
    mean_x = torch.mean(x, dim=0)[0]
    std_x = torch.std(x, dim=0)
    print('std is ', std_x)
    normal_x = (x - mean_x) / std_x
    return normal_x

def get_minmax_normal_data(x):
    min_x = torch.min(x, dim=0)[0]
    max_x = torch.max(x, dim=0)[0]
    normal_x = torch.div((x - min_x), (max_x - min_x))
    return normal_x, min_x, max_x

def normalize_mean_variance(data):
    mean = torch.mean(data, dim=0)
    std = torch.std(data, dim=0)
    normalized_data = (data - mean) / std
    return normalized_data, mean, std

def get_data_loader(inputs, targets):
    dataset = MyDataset(inputs, targets)

    total_size = len(dataset)
    train_ratio = 0.8
    val_ratio = 0.15
    test_ratio = 0.05

    # 计算每个数据集的大小
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader

def d_minmax_normal(x, min_val, max_val):
    # 判断输入类型，并选择适当的操作
    if isinstance(x, torch.Tensor):
        dnormal_x = torch.mul(x, (max_val - min_val)) + min_val
    elif isinstance(x, np.ndarray):
        if isinstance(min_val, torch.Tensor):
            min_val = min_val.cpu().numpy()
        elif isinstance(min_val, np.ndarray):
            min_val = min_val
        else:
            raise TypeError("min_val must be either a torch.Tensor or numpy.ndarray")

        if isinstance(max_val, torch.Tensor):
            max_val = max_val.cpu().numpy()
        elif isinstance(max_val, np.ndarray):
            max_val = max_val
        else:
            raise TypeError("max_val must be either a torch.Tensor or numpy.ndarray")

        dnormal_x = np.multiply(x, (max_val - min_val)) + min_val
    else:
        raise TypeError("Input must be either a torch.Tensor or numpy.ndarray")
    return dnormal_x


def d_zscore_normal(normalized_data, means, stds):
    denormalized_data = normalized_data * stds + means
    return denormalized_data

def one_hot_encoder(id_list):
    """
    将ID转化为按排序后的独热编码
    
    参数:
    id_list (list): 待编码的ID
    
    返回:
    numpy数组: 表示独热编码的数组
    """
    # 获取唯一的id并进行排序
    unique_ids = np.unique(id_list)
    sorted_ids = np.sort(unique_ids)  # 获取排序后的id列表
    
    # 构建原始id到新排序号的映射
    id_mapping = {original_id: new_id for new_id, original_id in enumerate(sorted_ids)}
    
    # 初始化独热编码数组
    data_num = len(id_list)
    num_unique_ids = len(sorted_ids)
    encoded = np.zeros((data_num, num_unique_ids))
    
    # 进行独热编码
    for i, original_id in enumerate(id_list):
        sorted_id = id_mapping[original_id]
        encoded[i][sorted_id] = 1  # 在对应的位置置1
    
    return encoded

def one_hot_decoder(id_list, onehot_code):
    """
    将onehot code转化为原始数据
    
    参数:
    id_list (list): 原始数据集ID
    onehot_code (numpy array): 需要解码的独热编码
    
    返回:
    numpy数组: 解码后的原始ID列表
    """
    # 获取唯一的id并进行排序
    unique_ids = np.unique(id_list)
    sorted_ids = np.sort(unique_ids)  # 获取排序后的id列表
    
    # 初始化一个空的decode_list用于存储解码后的ID
    decode_list = []
    
    # 对每一行的独热编码进行解码
    for onehot in onehot_code:
        # 获取值为1的位置
        index = np.argmax(onehot)  # 找到独热编码中1的位置
        decode_list.append(sorted_ids[index])  # 根据索引找到对应的原始id
    
    return np.array(decode_list)

class Material_Net(nn.Module):
    def __init__(self, mechanical_paras_layers, material_classes):
        super(Material_Net, self).__init__()
        self.hidden1 = nn.Linear(mechanical_paras_layers, 2000)
        self.hidden2 = nn.Linear(2000, 2000)
        self.hidden3 = nn.Linear(2000, 200)
        self.predict = nn.Linear(200, material_classes)

    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.hidden3(x)
        x = F.relu(x)
        predict = self.predict(x)
        return predict

class Type_Net(nn.Module):
    def __init__(self, mechanical_paras_layers, type_classes):
        super(Type_Net, self).__init__()
        self.hidden1 = nn.Linear(mechanical_paras_layers, 2000)
        self.hidden2 = nn.Linear(2000, 2000)
        self.hidden3 = nn.Linear(2000, 200)
        self.predict = nn.Linear(200, type_classes)

    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.hidden3(x)
        x = F.relu(x)
        predict = self.predict(x)
        return predict


class Regression_Net(nn.Module):
    def __init__(self, mechanical_paras_layers, geometry_para_layers, material_classes, type_classes):
        """耦合力学性能参数与晶胞制造方式，完成对

        Args:
            mechanical_paras_layers (_type_): _description_
            geometry_para_layers (_type_): _description_
            material_classes (_type_): _description_
        """
        super(Regression_Net, self).__init__()
        self.hidden1 = nn.Linear(mechanical_paras_layers + material_classes + type_classes, 2000)
        self.hidden2 = nn.Linear(2000, 2000)
        self.hidden3 = nn.Linear(2000, 500)
        self.predict = nn.Linear(500, geometry_para_layers)
        self.material_net = Material_Net(mechanical_paras_layers, material_classes)  # 创建 Material_Net 实例
        self.type_net = Type_Net(mechanical_paras_layers, type_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.material_net.load_state_dict(torch.load("ML/nn_model/pth_files/test_MaterialNet_model_best_01.pth"))
        self.material_net.eval()
        self.type_net.load_state_dict(torch.load("ML/nn_model/pth_files/test_TypeNet_model_best_01.pth"))
        self.type_net.eval()

    def forward(self, x_material_input):
        # 使用 Material_Net 的 forward 方法得到预测值
        with torch.no_grad():
            material_predict = self.material_net(x_material_input)
            type_predict = self.type_net(x_material_input)
        
        material_predict = F.softmax(material_predict, dim=1)
        type_predict = F.softmax(type_predict, dim=1)
        
        material_max_index = torch.argmax(material_predict, dim=1)
        type_max_index = torch.argmax(type_predict, dim=1)

        material_one_hot = F.one_hot(material_max_index, num_classes=num_material_classes)
        type_one_hot = F.one_hot(type_max_index, num_classes=num_type_classes)

        regrerssion_x_with_predict = torch.cat((x_material_input, material_one_hot, type_one_hot), dim=1)
        x = self.hidden1(regrerssion_x_with_predict)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.hidden3(x)
        x = F.relu(x)
        predict = self.predict(x)
        return predict

    
class forward_regression(nn.Module):
    def __init__(self, forward_input_dims, forward_output_dims):
        """_summary_

        Args:
            forward_input_dims (_type_): _description_
        """
        super(forward_regression, self).__init__()
        self.hidden1 = nn.Linear(forward_input_dims, 8000)
        self.hidden2 = nn.Linear(8000, 8000)
        self.hidden3 = nn.Linear(8000, 5000)
        self.hidden4 = nn.Linear(5000, 1000)
        self.predict = nn.Linear(1000, forward_output_dims)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float)
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.hidden3(x)
        x = F.relu(x)
        x = self.hidden4(x)
        x = F.relu(x)
        predict = self.predict(x)
        return predict

class CVAE(nn.Module):
    """Implementation of CVAE (Conditional Variational Auto-Encoder) for regression tasks"""
    
    def __init__(self, feature_size, cond_size, latent_size):
        super(CVAE, self).__init__()

        self.device = device
        self.forward_pth_path = forward_pth_path

        # Encoder
        self.fc1 = nn.Linear(feature_size + cond_size, 512)  # Concatenate features and continuous condition
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc_mu = nn.Linear(64, latent_size)  # Latent space mean
        self.fc_log_std = nn.Linear(64, latent_size)  # Latent space log std

        # Decoder
        self.fc5 = nn.Linear(latent_size + cond_size, 64)
        self.fc6 = nn.Linear(64, 128)
        self.fc7 = nn.Linear(128, 256)
        self.fc8 = nn.Linear(256, 512)
        self.fc9 = nn.Linear(512, feature_size)  # Output layer, same size as input

        # Forward regression network
        self.forward_net = forward_regression(forward_input_dim, forward_output_dim).to(device)

    def encode(self, x, y):
        """Concatenate input data (x) and condition (y) and pass through encoder"""
        cat = torch.cat([x, y], dim=1)
        h1 = F.relu(self.fc1(cat))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        h4 = F.relu(self.fc4(h3))
        mu = F.tanh(self.fc_mu(h4))
        log_std = F.tanh(self.fc_log_std(h4))
        return mu, log_std

    def decode(self, z, y):
        """Concatenate latent variables (z) and condition (y) and pass through decoder"""
        
        h1 = F.relu(self.fc5(torch.cat([z, y], dim=1)))
        h2 = F.relu(self.fc6(h1))
        h3 = F.relu(self.fc7(h2))
        h4 = F.relu(self.fc8(h3))
        recon = F.tanh(self.fc9(h4))
        # # Split the output into continuous and one-hot parts
        # recon_continuous = recon[:, :-3]  # Continuous outputs (non-one-hot encoded part)
        
        # # Apply sigmoid to the one-hot part
        # recon_one_hot_prob = torch.sigmoid(recon[:, -3:])  # Sigmoid output for one-hot encoding
        
        # # Convert probabilities to one-hot encoding by taking argmax
        # recon_one_hot = torch.zeros_like(recon_one_hot_prob)
        # recon_one_hot[torch.arange(recon_one_hot.size(0)), recon_one_hot_prob.argmax(dim=1)] = 1

        # # Concatenate continuous and one-hot encoded outputs back together
        # recon = torch.cat((recon_continuous, recon_one_hot), dim=1)
        return recon

    def reparametrize(self, mu, log_std):
        """Reparameterization trick: sampling latent variable z"""
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def predict_label(self, recon):
        """Predict the label based on the reconstruction"""
        material_class = torch.full((recon.shape[0], 1), 0, device=recon.device).to(self.device)
        recon_with_class = torch.cat([recon, material_class], dim=1).to(self.device)

        # Load forward network and make prediction
        self.forward_net.load_state_dict(torch.load(self.forward_pth_path))
        self.forward_net.eval()
        with torch.no_grad():
            prediction = self.forward_net(recon)
        return prediction

    def forward(self, x, y):
        """Forward pass: encode, reparameterize, and decode"""
        mu, log_std = self.encode(x, y)
        z = self.reparametrize(mu, log_std)
        recon = self.decode(z, y)
        pred = self.predict_label(recon)
        return recon, mu, log_std, pred

    def loss_function_with_class(self, recon, x, mu, log_std, target_class, beta=1.0):
        """Loss function combining reconstruction loss (continuous + classification) and KL divergence"""
        
        # 1. 分离 recon 中的连续值部分和 one-hot 编码的分类结果
        recon_continuous = recon[:, :-3]  # 拟合数据部分
        recon_one_hot = recon[:, -3:]  # 独热编码部分
        x_continues = x[:, :-3]  # 原始的连续输入部分

        # 2. 计算回归任务的重构损失（MSE）
        recon_loss_continuous = F.mse_loss(recon_continuous, x_continues, reduction="mean")

        # 3. 计算分类任务的交叉熵损失
        # 交叉熵损失需要原始的 logits 输入，而非经过 sigmoid 后的概率
        target_class_indices = target_class.argmax(dim=1)  # 将 one-hot 编码转为类别索引
        classification_loss = F.cross_entropy(recon_one_hot, target_class_indices, reduction="mean")

        # 4. 计算 KL 散度损失
        kl_loss = -0.5 * torch.sum(1 + 2 * log_std - mu.pow(2) - torch.exp(2 * log_std))

        # 5. 合并损失
        recon_loss = recon_loss_continuous + classification_loss
        total_loss = kl_loss * beta + recon_loss

        # print(f'KL loss: {kl_loss.item()}, recon loss (continuous + classification): {recon_loss.item()}, classification_loss: {classification_loss.item()}')
        return total_loss

    def loss_function_without_class(self, recon, x, mu, log_std, pred, y, beta=1.0):
        """Loss function for regression tasks without classification"""
        
        recon_loss_continuous = F.mse_loss(recon, x, reduction="mean")
        pred_loss = F.mse_loss(pred, y, reduction="mean")
        kl_loss = -0.5 * torch.sum(1 + 2 * log_std - mu.pow(2) - torch.exp(2 * log_std))
        # print("\nKL Loss:{}, Recon Loss:{}, Pre loss:{}".format(kl_loss, recon_loss_continuous, pred_loss) )
        total_loss = kl_loss * beta + recon_loss_continuous + 0
        return total_loss, kl_loss, recon_loss_continuous
 
 # Generator model
class Generator(nn.Module):
    def __init__(self, z_dim, y_dim, x_dim):
        super(Generator, self).__init__()
        self.fc = nn.Linear(z_dim + y_dim, 4096)        
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.predict = nn.Linear(256, x_dim)

    def forward(self, z, y):
        x = torch.cat([z, y], dim=1)
        x = F.relu(self.fc(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        x = torch.tanh(self.predict(x))
        return x

 # Discriminator model
class Discriminator(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Linear(x_dim + y_dim, 4096)        
        self.fc2 = nn.Linear(4096, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.predict = nn.Linear(256, 1)
        
    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = F.relu(self.fc(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        x = torch.sigmoid(self.predict(x))
        return x

# Regressor model
class Regressor(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Regressor, self).__init__()
        self.fc = nn.Linear(x_dim, 512)        
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.predict_x = nn.Linear(64, x_dim)
        self.predict_y = nn.Linear(64, y_dim)

    def forward(self, x):
        # x = torch.cat((x, d), dim=1)  # dim=1 表示在列维度上拼接
        x = F.relu(self.fc(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        latent_x = torch.tanh(self.predict_x(x))
        y = torch.tanh(self.predict_y(x))
        return latent_x, y

def read_config(json_path):
    with open(json_path) as json_file:
        config = json.load(json_file)
    return config
 
 
 
 
json_path = 'E:/01_Graduate_projects\Cellular_structures\Multi-functional_design\Code_Project\ML\ml.json'
config = read_config(json_path)

pth_save_addr = config["pth_address"]
relative_path = config["relative_address_baee"]
forward_pth_path = os.path.join(pth_save_addr, 'Forward_regression.pth')
best_g_model_path = os.path.join(pth_save_addr, 'best_generator.pth')
best_d_model_path = os.path.join(pth_save_addr, 'best_discriminator.pth')
best_r_model_path = os.path.join(pth_save_addr, 'best_regressor.pth')
forward_data_path = os.path.join(relative_path, 'm_results3.csv')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = np.genfromtxt(forward_data_path, delimiter=',', skip_header=1)


forward_input = data[:, 1:6]
forward_output = data[:,-2::]

forward_input = torch.tensor(forward_input).to(device)
forward_output = torch.tensor(forward_output).to(device)


normalized_forward_input, min_forward, max_forward = get_minmax_normal_data(forward_input)
normalized_forward_output, min_forward_output, max_forward_output = get_minmax_normal_data(forward_output)

forward_input_dim = normalized_forward_input.shape[1]
forward_output_dim = normalized_forward_output.shape[1]

forward_train, forward_val, forward_test = get_data_loader(normalized_forward_input, normalized_forward_output)

cellular_gan_output = forward_output
cellular_gan_input = forward_input

normalized_cellular_gan_input, min_gan_cellular, max_gan_cellular = get_minmax_normal_data(cellular_gan_input)
normalized_cellular_gan_output, min_cellular_gan_output, max_cellular_gan_output = get_minmax_normal_data(cellular_gan_output)

cellular_gan_input_dim = normalized_cellular_gan_input.shape[1]
cellular_gan_output_dim = normalized_cellular_gan_output.shape[1]

cellular_gan_train, cellular_gan_val, cellular_gan_test = get_data_loader(normalized_cellular_gan_input, normalized_cellular_gan_output)
