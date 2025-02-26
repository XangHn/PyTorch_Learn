import torch
import torchvision
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

dataset = torchvision.datasets.CIFAR10(root='../dataset', train=False, download=True, transform=transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool(input)
        return output


model = Model()

writer = SummaryWriter("../logs")

step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_image("input", imgs, step, dataformats="NCHW")
    output = model(imgs)
    writer.add_image("output", output, step, dataformats="NCHW")
    step += 1

writer.close()
