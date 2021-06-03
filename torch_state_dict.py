import torch

class TheModelClass(torch.nn.Module):
    def __init__(self):
        super(TheModelClass, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(-1, 16 * 5 * 5) # 하나로 펼쳐야 Fully connected layer에 입력 가능
        x = torch.nn.functional.relu(self.fc1)
        x = torch.nn.functional.relu(self.fc2)
        x = self.fc3(x)
        return x

model = TheModelClass()

optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)

# 모델의 state_dict 출력
print("Model's state_dict : ")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size()) # 각 layer의 size
    # print(param_tensor, "\t", model.state_dict()[param_tensor]) # 실제 값 출력


# opimizer의 state_dict 출력
print("Optimizer'sd state_dict : ")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])