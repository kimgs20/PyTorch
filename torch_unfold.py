"""
unfold(dimension, size, step)

원본 tensor의 모든 slices를 포함한 tensor를 반환함

몇번 째 차원을 어느 size로 몇 step만큼 띄어서 할 것인지

"""
import torch
x = torch.arange(1., 8)
print(x)

a = x.unfold(0, 3, 1)
b = x.unfold(0, 2, 2)
print(a)
print(b)