import torch

x = torch.ones(2, requires_grad=True)
y = 2*x
z = 3 + x
print(z)  # tensor([4., 4.], grad_fn=<AddBackward0>)

x = torch.ones(2, requires_grad=True)
y = 2*x
z = 3 + x.detach()  # new way
print(z)  # tensor([4., 4.])


x = torch.ones(2, requires_grad=True)
y = 2*x
z = 3 + x.data  # old way
print(z)  # tensor([4., 4.])

# x.detach() is a way to remove requires_grad and what you get is a new detached tensor
# tensor.detach() creates a tensor that shares storage with tensor that does not require grad.
# It detaches the output from the computational graph.
# So no gradient will be backpropagated along this variable.
