import numpy as np

# 指定 .npy 文件的路径
# file_path = "data/trajectories/samples_panda.npy"

# # 读取 .npy 文件
# data = np.load(file_path)
# print(data.shape[0])
# 查看数据内容
# for traj in data:
    # print(traj)


# # 定义 .dat 文件的路径
# dat_file_path = "data/trajectories/samples5000.dat"  # 替换为你的 .dat 文件路径
# npy_file_path = "data/trajectories/samples_panda5000.npy"  # 替换为目标 .npy 文件路径

# # 读取 .dat 文件数据
# # 假设数据是以空格或逗号分隔的数值
# try:
#     data = np.loadtxt(dat_file_path, delimiter=None)  # 根据需要设置 delimiter，例如 ',' 或 ' ' (默认自动检测)
    
#     # 保存为 .npy 文件
#     np.save(npy_file_path, data)
#     print(f"数据成功从 {dat_file_path} 转换为 {npy_file_path}")
# except Exception as e:
#     print(f"转换失败: {e}")



import numpy as np

def quaternion_angle_difference(q1, q2):
    # 确保四元数是单位四元数（长度为1）
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # 计算四元数之间的点积
    dot_product = np.dot(q1, q2)
    
    # 处理数值误差，确保点积在 [-1, 1] 范围内
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # 计算夹角（弧度）
    angle_rad = 2 * np.arccos(np.abs(dot_product))
    
    # 转换为角度
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

# 示例
q1 = [0.7071, 0, 0, 0.7071]  # 绕X轴旋转90度
q2 = [1, 0, 0, 0]  # 无旋转
angle_diff = quaternion_angle_difference(q1, q2)
print(f"旋转角度差异：{angle_diff}°")