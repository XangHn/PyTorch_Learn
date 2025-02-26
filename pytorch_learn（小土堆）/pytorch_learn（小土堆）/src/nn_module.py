import torch
from torch import nn  # 导入PyTorch的神经网络模块

# 定义一个自定义的神经网络模块，继承自nn.Module
class SimpleAddModule(nn.Module):
    def __init__(self):
        super().__init__()  # 调用父类的构造函数，初始化模块

    # 定义前向传播函数，input是输入张量
    def forward(self, input_tensor):
        # 对输入张量进行加1操作，并将结果赋值给output_tensor
        output_tensor = input_tensor + 1
        return output_tensor  # 返回输出张量


# 创建SimpleAddModule的实例
simple_add_module = SimpleAddModule()

# 创建一个标量张量，值为1.0
input_value = torch.tensor(1.0)

# 将输入张量传入模块，进行前向传播计算
output_value = simple_add_module(input_value)

# 打印输出结果
print(output_value)