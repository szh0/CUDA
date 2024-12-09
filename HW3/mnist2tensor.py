import os
import numpy as np 
import torch
from torchvision import datasets
import mytensor

from torch import nn
from torch.nn import functional as F

# 下载MNIST数据集
train_dataset = datasets.MNIST('~/.pytorch/MNIST_data/', train=True, download=True)
test_dataset = datasets.MNIST('~/.pytorch/MNIST_data/', train=False, download=True)

# 将训练集和测试集转换为NumPy格式
train_data = np.array([np.array(train_dataset[i][0]) for i in range(len(train_dataset))])
train_labels = np.array([train_dataset[i][1] for i in range(len(train_dataset))])
test_data = np.array([np.array(test_dataset[i][0]) for i in range(len(test_dataset))])
test_labels = np.array([test_dataset[i][1] for i in range(len(test_dataset))])

print("train_data.shape:", train_labels.shape)

numpy_input = train_data[0]
input = mytensor.np2tensor(numpy_input)
print(numpy_input)
input.print_data()