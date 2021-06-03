import torch

# tensor의 크기나 모양을 변경하는 것: torch.view

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8) # -1의 의미는 다른 차원에 맞추어 알아서 변경

print(x.size(), y.size(), z.size())

test = x.size(0)
print(test)