import torch

t = torch.FloatTensor([1, 2, 3])

c = torch.clamp(t, -2, 2)

d = torch.clamp(t, 2) # min 으로 해석함

print(c)

print(d)