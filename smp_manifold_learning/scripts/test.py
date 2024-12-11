import numpy as np

# 指定 .npy 文件的路径
file_path = "data/trajectories/samples_panda.npy"

# 读取 .npy 文件
data = np.load(file_path)
print(data.shape[0])
# 查看数据内容
# for traj in data:
    # print(traj)


# # 定义 .dat 文件的路径
# dat_file_path = "data/trajectories/samples_panda.dat"  # 替换为你的 .dat 文件路径
# npy_file_path = "data/trajectories/samples_panda.npy"  # 替换为目标 .npy 文件路径

# # 读取 .dat 文件数据
# # 假设数据是以空格或逗号分隔的数值
# try:
#     data = np.loadtxt(dat_file_path, delimiter=None)  # 根据需要设置 delimiter，例如 ',' 或 ' ' (默认自动检测)
    
#     # 保存为 .npy 文件
#     np.save(npy_file_path, data)
#     print(f"数据成功从 {dat_file_path} 转换为 {npy_file_path}")
# except Exception as e:
#     print(f"转换失败: {e}")