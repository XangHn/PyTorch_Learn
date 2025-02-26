import torch
import torchvision
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root='../dataset', train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(196608, 10)

    def forward(self, input):
        output = self.linear(input)
        return output



model = Model()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    output = torch.flatten(imgs)
    print(output.shape)
    output = model(output)
    print(output.shape)