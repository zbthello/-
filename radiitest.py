import torch
xyz = torch.tensor([[1., 1., 1.], [2, 2, 2]], device="cuda")  # 空间中有两个椭球
scaling = torch.tensor([[1., 2., 3.], [4., 5., 6.]], device="cuda")  # 缩放系数
rotation = torch.tensor([[1., 0., 0.], [1., 0., 0.]], device="cuda")  # 旋转系数
selected_pts_mask = torch.tensor([True, False], device="cuda")  # 选择要致密化的高斯椭球

stds = scaling[selected_pts_mask].repeat(2, 1)  # 根据旋转系数生成正太分布的标准差
means = torch.zeros((stds.size(0), 3), device="cuda")  # 根据缩放系数生成正态分布的均值
# samples张量中每个元素是从相互独立的正态分布中随机生成的。每个正态分布的均值和标准差对应着mean中的-个值和std中的一个值
samples = torch.normal(mean=means, std=stds)

rots = rotation[selected_pts_mask].repeat(2, 1, 1)
# torch.bmm 矩阵乘法
new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + xyz[selected_pts_mask].repeat(2, 1)
print(new_xyz)  # 将椭球(1, 1, 1)分裂为两个椭球(0.8083,  0.8083,  0.8083)|(-1.0404, -1.0404, -1.0404)






