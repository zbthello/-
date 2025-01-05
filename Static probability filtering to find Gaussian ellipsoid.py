import torch
from torch import nn
C0 = 0.28209479177387814
N = 2
def RGB2SH(rgb):
    return (rgb - 0.5) / C0
fused_objects = RGB2SH(torch.rand((N, 16), device="cuda"))
fused_objects = fused_objects[:, :, None]
print(fused_objects)
print(fused_objects.size())  # torch.Size([2, 16, 1])
objects_dc = nn.Parameter(fused_objects.transpose(1, 2).contiguous().requires_grad_(True))
print(objects_dc)
print(objects_dc.size())  # torch.Size([2, 1, 16])
classifier = torch.nn.Conv2d(16, 256, kernel_size=1)
classifier.cuda()
selected_obj_ids = torch.tensor(34).cuda()
logits3d = classifier(objects_dc.permute(2, 0, 1))
prob_obj3d = torch.softmax(logits3d, dim=0)
print(prob_obj3d)
print(prob_obj3d.size())  # torch.Size([256, 2, 1])
mask = prob_obj3d[selected_obj_ids, :, :] > 0.3
print(prob_obj3d[selected_obj_ids, :, :])  # tensor([[0.0042],[0.0041]], device='cuda:0', grad_fn=<SliceBackward0>)
print(prob_obj3d[selected_obj_ids, :, :].size())  # torch.Size([2, 1])
print(mask)  # tensor([[False],[False]], device='cuda:0')
print(mask.size())  # torch.Size([2, 1])
mask3d = mask.any(dim=0).squeeze()
