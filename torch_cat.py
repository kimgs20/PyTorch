import torch

x = torch.FloatTensor([[1, 2, 3], [3, 4, 6]])
y = torch.FloatTensor([[7, 8, 9], [10, 11, 12]])

print(x.size()) # 2, 3

a = torch.cat([x, y], dim=0) # 첫번째 차원인 행방향으로 cat
print(a)
print(a.size()) # 4, 3

b = torch.cat([x, y], dim=1) # 두번째 차원인 열방향으로 cat
print(b)
print(b.size()) # 2, 6

c = torch.cat([x, y])
print(c)
print(c.size())

a = torch.tensor([[0]])
print(a.size()) # 1, 1

b = torch.tensor([0])
print(b.size()) # 1