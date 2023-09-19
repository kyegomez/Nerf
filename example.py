import torch
from nerf.model import Nerf

model = Nerf()

x = torch.randn(1, 6)

output = model(x)

print(output)