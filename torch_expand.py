'''
torch.repeat(*sizes)
copy tensor

torch.expand(*sizes)
same as repeat but can use only 1-dimension
'''

import torch
'''
# torch.repeat(*sizes)
x = torch.tensor([1, 2, 3])
x_42 = x.repeat(4, 2)  # 4row 2column
print(x_42)
print(x_42.size())  # [4, 6]

x_421 = x.repeat(4, 2, 1)
print(x_421)
print(x_421.size())  # [4, 2, 3]
'''

# torch.expand(*sizes)
x = torch.tensor([[1], [2], [3]])
print(x.size())
print()

x_exp = x.expand(3, 1)
print(x_exp)
print()

x_exp = x.expand(3, 3)
print(x_exp)
print()

x_exp = x.expand(3, 4)  # if first dim is not 3, error occurs
print(x_exp)
print(x_exp.size())
print()

x_exp_mi = x.expand(-1, 4)  # -1 means not changin the size of that dimension
print(x_exp_mi)
print(x_exp_mi.size())
# x_exp == x_exp_mi'''