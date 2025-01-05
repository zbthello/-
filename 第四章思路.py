import torch
N = 1; H = 2; W = 2
K_2D = torch.randn((N, H, W), device="cuda")
classifier = torch.nn.Conv2d(N, N+1, kernel_size=1, device="cuda")
inputs = torch.unsqueeze(K_2D, dim=0)
print(inputs)
print(inputs.size())
outputs = classifier(inputs)
print(outputs)
print(outputs.size())

