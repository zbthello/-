import torch
import numpy as np

# 进行2D变换

# 平移  x`= x + t
xy_1 = torch.tensor([[1., 2.], [3., 4.]], device="cuda")  # 二维平面上的点
xy_t = torch.tensor([[0.4, 0.6], [0.4, 0.6]], device="cuda")  # 平移的距离
xy_2 = xy_1 + xy_t  # 平移操作
print(xy_2)  # tensor([[1.4000, 2.6000],[3.4000, 4.6000]], device='cuda:0')

# 欧式  x` = Rx + t
xy_3 = torch.tensor([[torch.sqrt(torch.tensor([3])), 1.]], device="cuda")  # 二维平面上的点
q = torch.tensor([(30 * np.pi) / 180], device="cuda")  # 30度通过公式((angle * np.pi) / 180)转换为弧度
R = torch.tensor([[torch.cos(q), -torch.sin(q)], [torch.sin(q), torch.cos(q)]], device="cuda")
t = torch.tensor([[0.4, 0.6]], device="cuda")  # 平移的距离
# torch.mm数学里的矩阵乘法，要求两个Tensor的维度满足矩阵乘法的要求.
xy_4 = torch.mm(xy_3, R) + t  # 旋转+平移 (sqrt(3)=1.7, 1) ===> (2, 0) ===>(2.4, 0.6)
print(xy_4)  # tensor([[2.4000, 0.6000]], device='cuda:0')


