import torch
import torch.nn as nn

# import flopth
from flopth import flopth

# define Model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(3, 3, kernel_size=3, padding=1)

    def forward(self, x1):
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.conv4(x1)
        return x1


# declare Model object
my_model = MyModel()

# Use input size
flops, params = flopth(my_model, in_size=((3, 224, 224),))
print(flops, params)

# # Or use input tensors
# dummy_inputs = torch.rand(1, 3, 224, 224)
# flops, params = flopth(my_model, inputs=(dummy_inputs,))
# print(flops, params)