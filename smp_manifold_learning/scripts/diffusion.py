import torch
import torch.nn as nn
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class DiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 128, 64]):
        super().__init__()
        
        # 网络架构（与之前代码相同）
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, input_dim))
        self.model = nn.Sequential(*layers)
        
        # 噪声调度参数
        self.beta_start = 0.0001
        self.beta_end = 0.02
        self.num_timesteps = 1000

    def forward_diffusion(self, x0):
        """前向扩散过程"""
        device = x0.device
        batch_size = x0.shape[0]
        
        # 生成噪声时间步
        timesteps = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        
        # 计算噪声强度
        betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps, device=device)
        alphas = 1 - betas
        cumulative_alphas = torch.cumprod(alphas, dim=0)
        
        # 添加噪声
        noise = torch.randn_like(x0)
        selected_alphas = cumulative_alphas[timesteps].view(-1, 1)
        noisy_x = torch.sqrt(selected_alphas) * x0 + torch.sqrt(1 - selected_alphas) * noise
        
        return noisy_x, noise, timesteps

    def reverse_diffusion(self, noisy_x, timestep):
        """逆向去噪过程"""
        predicted_noise = self.model(noisy_x)
        return predicted_noise

    def train_step(self, batch):
        """单步训练"""
        self.optimizer.zero_grad()
        
        noisy_x, true_noise, timesteps = self.forward_diffusion(batch)
        predicted_noise = self.reverse_diffusion(noisy_x, timesteps)
        
        loss = nn.MSELoss()(predicted_noise, true_noise)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


    def save_model(self, save_path='robotic_diffusion_model.pth'):
        """
        保存模型
        
        参数:
        - save_path: 模型保存路径
        """
        # 确保目录存在
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 保存模型状态字典和配置
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_dim': self.model[0].in_features,
            'hidden_dims': [layer.out_features for layer in self.model[:-1:3]],
            'beta_start': self.beta_start,
            'beta_end': self.beta_end,
            'num_timesteps': self.num_timesteps
        }, save_path)
        
        print(f"模型已保存到 {save_path}")

    @classmethod
    def load_model(cls, load_path='robotic_diffusion_model.pth'):
        """
        加载模型
        
        参数:
        - load_path: 模型加载路径
        
        返回:
        - 重建的模型实例
        """
        # 加载检查点
        checkpoint = torch.load(load_path)
        
        # 使用保存的配置重建模型
        model = cls(
            input_dim=checkpoint['input_dim'], 
            hidden_dims=checkpoint['hidden_dims']
        )
        
        # 恢复模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 恢复噪声调度参数
        model.beta_start = checkpoint['beta_start']
        model.beta_end = checkpoint['beta_end']
        model.num_timesteps = checkpoint['num_timesteps']
        
        print(f"从 {load_path} 加载模型成功")
        return model

    def generate_samples(self, num_samples, num_inference_steps=1000):
        """
        生成新的关节角度样本
        
        参数:
        - num_samples: 生成样本数量
        - num_inference_steps: 去噪步数
        
        返回:
        - 生成的样本
        """
        self.eval()  # 设置为评估模式
        device = next(self.parameters()).device
        
        # 初始随机噪声
        # x = torch.randn(num_samples, self.model[0].in_features, device=device)
        # print("x: ",x)
        x = torch.tensor(np.array([[2.89727, 0.731899, -0.36857 ,-0.771544 ,-2.89723 ,1.66956 ,2.85572]]), dtype=torch.float32)
        # 逆向去噪过程
        betas = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps, device=device)
        alphas = 1 - betas
        sqrt_alphas = torch.sqrt(alphas)
        sqrt_one_minus_alphas = torch.sqrt(1 - alphas)
        
        for t in reversed(range(num_inference_steps)):
            z = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
            
            # 预测噪声
            with torch.no_grad():
                predicted_noise = self.model(x)
            
            # 去噪
            x = (x - predicted_noise * sqrt_one_minus_alphas[t]) / sqrt_alphas[t] + \
                betas[t] * z
        
        return x.cpu().numpy()

# 使用示例
def main():

    data = np.load("../data/trajectories/samples_panda5000.npy")

    # 训练后保存模型
    input_dim = 7  # 假设机械臂有6个关节
    # model = train_diffusion_model(data)
    
    # 假设已经训练完成
    # 保存模型
    # model.save_model('robotic_arm_diffusion_model.pth')
    
    # 加载模型
    loaded_model = DiffusionModel.load_model('robotic_arm_diffusion_model.pth')
    
    # # 生成新样本
    new_joint_angles = loaded_model.generate_samples(num_samples=10)
    # print("生成的关节角度样本:")
    print(new_joint_angles)


def train_diffusion_model(data, num_epochs=100, batch_size=64):
    """训练主函数"""
    # 数据准备
    data_tensor = torch.FloatTensor(data)
    dataset = TensorDataset(data_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 模型初始化
    input_dim = data.shape[1]
    model = DiffusionModel(input_dim)
    model.optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 训练循环
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            loss = model.train_step(batch[0])
            total_loss += loss
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader)}")
    
    return model

if __name__ == '__main__':
    main()