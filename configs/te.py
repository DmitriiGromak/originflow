from tqdm import tqdm
import time
import torch
# 创建一个示例列表
data = range(100)
# Set-up time
ts = torch.linspace(
  1e-2, 1.0, 500)
t_1 = ts[0]


clean_traj = []


# 使用 tqdm 仅显示进度条而不会刷新控制台
for item in tqdm(data, leave=True, ):
    # 模拟处理
    time.sleep(0.1)
