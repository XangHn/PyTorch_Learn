import torch
import torchvision
from torch import nn
from model_save import *

# 方式1 -> 保存方式1，加载模型
# model_1 = torch.load("vgg16_method1.pth")
# print(model_1)


# 方式2 -> 保存方式1，加载模型
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
# model_2 = torch.load("vgg16_method2.pth")
# print(model_2)
# print(vgg16)






# 陷阱
# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x

# model = Model()   不需要写这一步
model = torch.load("model_method1.pth")
print(model)
