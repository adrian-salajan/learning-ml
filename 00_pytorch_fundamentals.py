import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

scalar = torch.Tensor(7)

print(scalar)
print(scalar.ndim)

vector = torch.tensor([3, 4])
print(vector)

t123 = torch.Tensor([[[4, 4, 4], [5, 5, 5]]])
print(t123.ndim) #3
print(t123.shape)

t321 = torch.Tensor([[[4], [0]], [[5], [0]], [[6], [0]]])
print(t321.ndim)
print(t321.shape)

image = torch.rand(3, 6, 6)
print(image)

image2 = torch.rand(9, 4, 2)
print(image2)

print(torch.arange(1, 10, 2).shape)

print(torch.zeros_like(image))