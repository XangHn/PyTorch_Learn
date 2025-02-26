import torch
from torch import nn
from torch.nn import ReLU

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu_1 = ReLU()

    def forward(self, input):
        output = self.relu_1(input)
        return output

model = Model()
output = model(input)
print(output)