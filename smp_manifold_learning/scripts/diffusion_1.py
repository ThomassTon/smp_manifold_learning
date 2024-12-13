import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 数据加载
# 假设已经有一个 20000 x N 的 NumPy 数组 joint_angles，N 是机械臂的关节数。
# 将数据标准化。
data = np.load("../data/trajectories/samples_panda5000.npy")  # 加载数据
mean = data.mean(axis=0)
std = data.std(axis=0)
data_normalized = (data - mean) / (std + 1e-8)

# 转换为 PyTorch 数据格式
data_tensor = torch.tensor(data_normalized, dtype=torch.float32)
dataset = TensorDataset(data_tensor)
data_loader = DataLoader(dataset, batch_size=128, shuffle=True)

# 定义扩散模型
class DiffusionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DiffusionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, t):
        # 将时间步 t 作为输入的一部分
        t_embedding = torch.sin(t.unsqueeze(-1) * torch.linspace(1, 10, x.size(-1)).to(x.device))
        x_t = torch.cat([x, t_embedding], dim=-1)
        return self.net(x_t)

# 训练函数
def train_diffusion(model, optimizer, dataloader, num_steps, timesteps, device):
    model.train()
    mse_loss = nn.MSELoss()

    for step in range(num_steps):
        for batch in dataloader:
            x = batch[0].to(device)

            # 生成随机时间步 t 和噪声
            t = torch.randint(0, timesteps, (x.size(0),), device=device, dtype=torch.long)
            noise = torch.randn_like(x).to(device)

            # 添加噪声
            alpha = torch.cos(torch.linspace(0, np.pi / 2, timesteps).to(device))
            alpha_t = alpha[t].unsqueeze(-1)
            noisy_x = alpha_t * x + (1 - alpha_t) * noise

            # 预测噪声
            predicted_noise = model(noisy_x, t.float() / timesteps)

            # 损失计算并反向传播
            loss = mse_loss(predicted_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}/{num_steps}, Loss: {loss.item():.4f}")

def train():
    # 初始化模型和优化器
    input_dim = data.shape[1]  # 输入维度（关节角度数）
    hidden_dim = 128
    num_timesteps = 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffusionModel(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 开始训练
    train_diffusion(model, optimizer, data_loader, num_steps=5000, timesteps=num_timesteps, device=device)

    # 保存模型
    torch.save(model.state_dict(), "diffusion_model.pth")




if __name__ =='__main__':
    # 加载训练好的模型
    device = 'cuda'
    model = DiffusionModel(input_dim=7,hidden_dim = 128)  # 与训练时的模型架构一致
    model.load_state_dict(torch.load('diffusion_model.pth'))
    model.eval()  # 将模型设置为评估模式

    new_data = np.array([0.0533224, -0.10808, -2.10303, -0.0698, 2.49767, 2.98619, 1.90496])  # 示例数据，实际输入应为新的机械臂关节角度

    # 将数据转换为Tensor
    new_data_tensor = torch.tensor(new_data, dtype=torch.float32).to(device)

    # 使用模型进行预测
    with torch.no_grad():  # 不需要计算梯度
        predicted_angles = model(new_data_tensor)

    # 打印预测结果
    print(predicted_angles)