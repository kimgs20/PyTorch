"""
squeeze 함수는 차원이 1인 차원을 제거해준다.
따로 차원을 설정하지 않으면 1인 차원을 모두 제거

주의할 점은 생각치도 못하게 batch가 1일 때 batch차원도 없애버리는 
불상사가 발생할 수있다. 그래서 validation단계에서 오류가 날 수 있기 
때문에 주의해서 사용해야 한다.

unsqueeze함수는 squeeze함수의 반대로 1인 차원을 생성하는 함수이다. 
그래서 어느 차원에 1인 차원을 생성할 지 꼭 지정해주어야한다.
"""

import torch

x = torch.rand(2, 1, 3, 3)
print(x)
print(x.size())

y = torch.squeeze(x)
print(y)
print(y.size())

x = torch.rand(3, 2, 2)
y = x.unsqueeze(dim=1)
print(x)
print(x.size())
print(y)
print(y.size())