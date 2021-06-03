import torch

"""
docs:
torch.gather(input, dim, index, out=None, sparse_grad=False)

"Gathers values along an axis specified by dim."

torch.gather은 input tensor가 입력으로 추어지고,
차원 dim을 따라 각 행에서 value를 취해서 새로운 tensor를 만든다.

왜 필요하냐?
어떤 tensor가 있을 때 꽤나 복잡한 indexing을 하기 위해

# torch.gather(input, dim, index, out=None, sparse_grad=False)
# input = input tensor
# dim = dimension along tensor
# index = tensor with indices of values to collect
"""
tensor = torch.FloatTensor([[0, 1, 2, 3, 4, 5],
                        [10, 11, 12, 13, 14, 15],
                        [20, 21, 22, 23, 24, 25],
                        [30, 31, 32, 33, 34, 35]])

# 2, 12, 22, 32 를 모을 때 -> 쉽게 된다
# column_tensor = tensor[:,2]
# print(column_tensor)

# 2, 14, 22, 30 (각 행 마다 특정 요소)을 모을 때 잘못하는 경우
# indices = torch.LongTensor([2, 4, 2, 0])
# arb_tensor = tensor[:, indices]
# print(arb_tensor) # 원하는 대로 잘 안됨

# 이런 문제로 인해 gather이 필요함

gathered_tensor = torch.gather(tensor, 1, torch.LongTensor([[4, 4],[1, 2],[2, 0],[0, 5]])) # input, dim, index
print("gathered_tensor: ", gathered_tensor)
print()

# gathered_tensor2 = tensor.gather(1, torch.LongTensor([[0, 2],[1, 1],[2, 0],[3, 0]]))
# print("gathered_tensor2: ", gathered_tensor2)

# pytorch DQN에서 일어나는 것
# policy_net_output = torch.tensor([[0.3, 0.2], [0.6, 0.8], [0.1, 0.7], [0.15, 0.4], [0.9, 0.5]])
# action_batch = torch.tensor([[0], [1], [0], [1], [1]])

# state_action_values = policy_net_output.gather(1, action_batch)
# print(state_action_values)