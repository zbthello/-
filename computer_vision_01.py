import torch

xy_1 = torch.tensor([[1., 2.], [3., 4.]], device="cuda")  # 二维平面上的点

# 进行2D变换

# 平移
xy_c = torch.tensor([[0.4, 0.6], [0.4, 0.6]], device="cuda")  # 平移的距离
xy_2 = xy_c + xy_1  # 平移操作
print(xy_2)  # tensor([[1.4000, 2.6000],[3.4000, 4.6000]], device='cuda:0')
print(xy_2.size())  # torch.Size([2, 2])
