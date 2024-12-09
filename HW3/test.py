import os
import numpy as np 
import torch
from torchvision import datasets
import mytensor

from torch import nn
from torch.nn import functional as F


# pytorch.Tensor <--> numpy
# numpy <--> mytensor.Tensor

def check(res, tar):
    if tar.ndim == 0 or res.ndim == 0:
        if tar.ndim == 0 : 
            tar_expand = np.expand_dims(res, axis=0)
        else :
            tar_expand = tar

        if res.ndim == 0 :
            res_expand = np.expand_dims(tar, axis=0)
        else :
            res_expand = res
        assert tar_expand == res_expand
    else :
        assert res.shape == tar.shape
        assert np.allclose(res, tar, atol=1e-5)


# def test_fc(shape):
#     input = torch.randn(shape['batchsize'], shape['in_features'], 
#                         requires_grad=True)
#     weight = torch.randn(shape['out_features'], shape['in_features'], 
#                          requires_grad=True)
#     bias = torch.randn(shape['out_features'], requires_grad=True)

#     # forward 
#     # torch.nn
#     output = F.linear(input, weight, bias)
#     print(output.detach().numpy())
#     # mytensor
#     t_input = mytensor.np2tensor(input.detach().numpy())
#     trans_weight = weight.transpose(0, 1)
#     t_weight = mytensor.np2tensor(trans_weight.detach().numpy())
#     t_bias = mytensor.np2tensor(bias.detach().numpy())
#     t_output = mytensor.Tensor(output.detach().numpy().shape,"gpu") 

#     mytensor.fc_forward(t_input, t_output, t_weight, t_bias)
#     t_output.print_data()


#     # backward
#     # torch.nn
#     grad_output = torch.randn(shape['batchsize'], shape['out_features'],
#                               requires_grad=False)
#     output.backward(grad_output)

def test_conv(shape):
    out_height = (shape['in_height'] + 2*shape['padding'] - shape['kernel_size']) // shape['stride'] + 1
    out_width = (shape['in_width'] + 2*shape['padding'] - shape['kernel_size']) // shape['stride'] + 1
    input = torch.randn(shape['batchsize'], shape['in_channels'], shape['in_height'], shape['in_width'], 
                        requires_grad=True)
    kernel = torch.randn(shape['out_channels'], shape['in_channels'], shape['kernel_size'], shape['kernel_size'], 
                         requires_grad=True)
    bias = torch.randn(shape['out_channels'], requires_grad=True)

    # forward 
    # torch.nn
    output = F.conv2d(input, kernel, bias, stride=shape['stride'], padding=shape['padding'])
    
    # mytensor
    t_input = mytensor.np2tensor(input.detach().numpy())
    t_kernel = mytensor.np2tensor(kernel.detach().numpy())
    t_bias = mytensor.np2tensor(bias.detach().numpy())
    t_output = mytensor.Tensor(output.detach().numpy().shape,"gpu") 

    mytensor.conv_forward(t_input, t_output, t_kernel, t_bias, 
                          shape['kernel_size'], shape['stride'], shape['padding'])
    check(mytensor.tensor2np(t_output), output.detach().numpy())

    # backward
    # torch.nn
    grad_output = torch.randn(shape['batchsize'], shape['out_channels'], out_height, out_width,
                              requires_grad=False)
    output.backward(grad_output)

    # mytensor
    # intialize grad_input, grad_kernel, grad_bias (initialized to zero !)
    t_grad_input = mytensor.Tensor(input.grad.numpy().shape,"gpu")
    mytensor.zeros_init(t_grad_input)
    t_grad_output = mytensor.np2tensor(grad_output.detach().numpy())
    t_grad_kernel = mytensor.Tensor(kernel.grad.numpy().shape,"gpu")
    mytensor.zeros_init(t_grad_kernel)
    t_grad_bias = mytensor.Tensor(bias.grad.numpy().shape,"gpu")
    mytensor.zeros_init(t_grad_bias)

    mytensor.conv_backward(t_input, t_output, t_kernel, t_bias, 
                        t_grad_input, t_grad_output, t_grad_kernel, t_grad_bias,
                        shape['kernel_size'], shape['stride'], shape['padding'])

    # check
    check(mytensor.tensor2np(t_grad_output), grad_output.detach().numpy())
    check(mytensor.tensor2np(t_grad_input), input.grad.numpy())
    check(mytensor.tensor2np(t_grad_kernel), kernel.grad.numpy())
    check(mytensor.tensor2np(t_grad_bias), bias.grad.numpy())
  
def test_max_pool(shape):
    out_height = (shape['in_height'] + 2*shape['padding'] - shape['kernel_size']) // shape['stride'] + 1
    out_width = (shape['in_width'] + 2*shape['padding'] - shape['kernel_size']) // shape['stride'] + 1
    input = torch.randn(shape['batchsize'], shape['in_channels'], shape['in_height'], shape['in_width'], 
                        requires_grad=True)

    # forward 
    # torch.nn
    output, mask = F.max_pool2d(input, kernel_size=shape['kernel_size'], stride=shape['stride'], padding=shape['padding'], return_indices=True)

    # mytensor
    t_input = mytensor.np2tensor(input.detach().numpy())
    t_output = mytensor.Tensor(output.detach().numpy().shape,"gpu") 
    t_mask = mytensor.Tensor(input.detach().numpy().shape,"gpu")
    mytensor.zeros_init(t_mask)

    mytensor.maxpool_forward(t_input, t_output, t_mask, shape['kernel_size'], shape['stride'], shape['padding'])
    check(mytensor.tensor2np(t_output), output.detach().numpy())


    # backward
    # torch.nn
    grad_output = torch.randn(shape['batchsize'], shape['in_channels'], out_height, out_width,
                              requires_grad=True)
    output.backward(grad_output)
    
    # mytensor
    t_grad_input = mytensor.Tensor(input.grad.numpy().shape,"gpu")
    mytensor.zeros_init(t_grad_input)

    t_grad_output = mytensor.np2tensor(grad_output.detach().numpy())
    mytensor.maxpool_backward(t_input, t_output, t_mask, t_grad_input, t_grad_output, shape['kernel_size'], shape['stride'], shape['padding'])

    # check(mytensor.tensor2np(t_grad_input), input.grad.numpy())
    check(mytensor.tensor2np(t_grad_output), grad_output.detach().numpy())

# and softmax
def test_celoss(shape):
    input = torch.randn(shape['batchsize'], shape['labels'], 
                        requires_grad=True)
    target = torch.randint(0, shape['labels'], (shape['batchsize'],))
    # forward
    # torch.nn
    loss = F.cross_entropy(input, target)

    # mytensor    
    t_input = mytensor.np2tensor(input.detach().numpy())
    t_output = mytensor.Tensor(input.detach().numpy().shape,"gpu")
    t_target = mytensor.np2tensor(target.detach().numpy())
    t_loss = mytensor.Tensor([1],"gpu")

    mytensor.softmax_forward(t_input, t_output)
    mytensor.cross_entropy_forward(t_output, t_target, t_loss)

    # check
    check(mytensor.tensor2np(t_loss), loss.detach().numpy())

    # backward
    # torch.nn
    loss.backward()
    
    # mytensor
    t_grad_input = mytensor.Tensor(input.grad.numpy().shape,"gpu")
    mytensor.cross_entropy_backward(t_input, t_output, t_target, t_loss, t_grad_input)

    # check
    check(mytensor.tensor2np(t_grad_input), input.grad.detach().numpy())

def main():
    # shape = {
    #     'batchsize': 10,
    #     'in_features': 28,
    #     'out_features': 15,
    #     'labels': 10,
    #     'in_channels': 3,
    #     'out_channels': 6,
    #     'in_height': 28,
    #     'in_width': 28,
    #     'stride': 1,
    #     'padding': 0,
    #     'kernel_size': 3,
    #     # maxpool: kernel_size = stride
    # }
    shape = {
        'batchsize': 1,
        'in_features': 28,
        'out_features': 15,
        'labels': 10,
        'in_channels': 1,
        'out_channels': 1,
        'in_height': 4,
        'in_width': 4,
        'stride': 1,
        'padding': 1,
        'kernel_size': 3,
        # maxpool: kernel_size = stride
    }

    test_celoss(shape)
    # test_fc(shape)
    test_conv(shape)
    test_max_pool(shape)


if __name__ == "__main__":
    main()
# input = torch.randn(config['batchsize'], config['labels'], 
#                         requires_grad=True)
# target = torch.from_numpy(train_labels[:config['batchsize']])
# # target = torch.randint(0, config['labels'], (config['batchsize'],))
# loss = F.cross_entropy(input, target)
# print(loss)

# loss.backward()

# output = F.linear(input, weight, bias)
# output.backward(grad_output)

# input = mytensor.Tensor([10,3],"gpu")
# output = mytensor.Tensor([10,3],"gpu")

# mytensor.tensor_init(input)
# mytensor.tensor_init(output)

# input.print_data()

# mytensor.softmax_forward(input,output)

# output.print_data()