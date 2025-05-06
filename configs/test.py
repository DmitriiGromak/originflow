import torch


fixed_mask=torch.tensor([0,1,1,0,0,1])
positions = torch.nonzero(fixed_mask == 1).squeeze()
# 转换为列表
positions_list = positions.tolist()
# 转换为逗号分隔的字符串
positions_str = ', '.join(map(str, positions_list[-1]))