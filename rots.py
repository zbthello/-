import torch
rots = torch.zeros((2, 4), device="cuda")
rots[:, 0] = 1
print(rots)