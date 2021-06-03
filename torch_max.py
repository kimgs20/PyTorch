import torch

x = torch.tensor([[-1.2148, -1.3570, -1.2952, -1.1476]])
print(x.max(1)[1])
print(x.max(1)[1].view(1, 1))


'''
# a = torch.randn(1, 3) # torch.Size([1, 3])
# a = torch.FloatTensor([1, 2, 3]) # torch.Size([3])
a = torch.FloatTensor([[1, 2, 3]]) # torch.Size([1, 3])
print('a', a)
print(type(a))
print(a.size())
print()

b = torch.max(a) # 최대값
print('b', b)
print(type(b))
print()

c = a.max(1) # max(1) 은 return_type.max 라는 class를 반환
print('c', c) # value와 index
print(type(c)) # <class 'torch.return_types.max'>
print()

d = a.max(1)[0] # [0]을 하면 value 만을 tensor로 받음
print('d', d)
print()

e = a.max(1)[1] # 이게 예제에서 사용한 문법
print("e", e) # 최대값의 위치 인덱스만을 tensor로 받음
print(e.size()) # 1
print()

f = a.max(1)[1].view(1,1)
print('f', f)
print(f.size()) # 1, 1
print()

g = f.view(1)
print('g', g)
print(f.size())
'''