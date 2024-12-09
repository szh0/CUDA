import argparse
import subprocess
import os
import json

import torch
from torch import nn
from torch.nn import functional as F

def to_str(x: torch.Tensor):
    x_str = x.detach().numpy().astype(str)
    x_str = ' '.join(x_str.flatten()) + '\n'
    return x_str

def get_args():
    parser = argparse.ArgumentParser(
        description="Generate random data and test modules.")
    parser.set_defaults(save=False, regenerate=False)
    parser.add_argument('-m', '--module', dest='module', type=str, choices=['fc', 'conv', 'maxpool', 'celoss'], required=True)
    parser.add_argument('-s', '--save', dest='save', action='store_true',
                        help="Save test data")
    parser.add_argument('-r', '--regenerate', dest='regenerate', action='store_true',
                        help="Regenerate test data")
    return parser.parse_args()

def get_config():
    with open('config.json', 'r', encoding='utf-8') as config_file:
        config = json.load(config_file)
    return config

def generate_fc(config):
    input = torch.randn(config['batchsize'], config['in_features'], 
                        requires_grad=True)
    weight = torch.randn(config['out_features'], config['in_features'], 
                         requires_grad=True)
    bias = torch.randn(config['out_features'], 
                       requires_grad=True)
    grad_output = torch.randn(config['batchsize'], config['out_features'], 
                              requires_grad=True)
    output = F.linear(input, weight, bias)
    output.backward(grad_output)
    with open('data/fc.txt', 'w') as f:
        f.write(' '.join(str(value) for value in config.values()) + '\n')
        f.write(to_str(input))
        f.write(to_str(output))
        f.write(to_str(weight.transpose(0, 1)))
        f.write(to_str(bias))
        f.write(to_str(grad_output))
        f.write(to_str(input.grad))
        f.write(to_str(weight.grad.transpose(0, 1)))
        f.write(to_str(bias.grad))

def generate_conv(config):
    height_out = (config['height'] + 2 * config['pad'] - config['ksize']) // config['stride'] + 1
    width_out = (config['width'] + 2 * config['pad'] - config['ksize']) // config['stride'] + 1
    input = torch.randn(config['batchsize'], config['channels_in'], config['height'], config['width'], 
                        requires_grad=True)
    kernel = torch.randn(config['channels_out'], config['channels_in'], config['ksize'], config['ksize'], 
                         requires_grad=True)
    bias = torch.randn(config['channels_out'], 
                       requires_grad=True)
    grad_output = torch.randn(config['batchsize'], config['channels_out'], height_out, width_out, 
                              requires_grad=True)
    output = F.conv2d(input, kernel, bias, config['stride'], config['pad'])
    output.backward(grad_output)
    with open('data/conv.txt', 'w') as f:
        f.write(' '.join(str(value) for value in config.values()) + '\n')
        f.write(to_str(input))
        f.write(to_str(output))
        f.write(to_str(kernel))
        f.write(to_str(bias))
        f.write(to_str(grad_output))
        f.write(to_str(input.grad))
        f.write(to_str(kernel.grad))
        f.write(to_str(bias.grad))

def generate_maxpool(config):
    height_out = (config['height'] + 2 * config['pad'] - config['ksize']) // config['stride'] + 1
    width_out = (config['width'] + 2 * config['pad'] - config['ksize']) // config['stride'] + 1
    input = torch.randn(config['batchsize'], config['channels'], config['height'], config['width'], 
                        requires_grad=True)
    grad_output = torch.randn(config['batchsize'], config['channels'], height_out, width_out, 
                              requires_grad=True)
    output, mask = F.max_pool2d(input, config['ksize'], config['stride'], config['pad'], return_indices=True)
    output.backward(grad_output)
    with open('data/maxpool.txt', 'w') as f:
        f.write(' '.join(str(value) if type(value) is int else '' for value in config.values()) + '\n')
        f.write(to_str(input))
        f.write(to_str(output))
        f.write(to_str(grad_output))
        f.write(to_str(input.grad))
        # f.write(to_str(mask))

def generate_celoss(config):
    input = torch.randn(config['batchsize'], config['labels'], 
                        requires_grad=True)
    target = torch.randint(0, config['labels'], (config['batchsize'],))
    loss = F.cross_entropy(input, target)
    loss.backward()
    with open('data/celoss.txt', 'w') as f:
        f.write(' '.join(str(value) for value in config.values()) + '\n')
        f.write(to_str(input))
        f.write(to_str(target))
        f.write(to_str(loss))
        f.write(to_str(input.grad))

def generate(module: str, config: dict):
    func = globals().get(f'generate_{module}')
    if func and module in config:
        func(config[module])
    else:
        print(f'Unknown module: {module}')

def main():
    args = get_args()
    config = get_config()

    if not os.path.exists('data'):
        os.mkdir('data')
    if args.regenerate or not os.path.exists(f'data/{args.module}.txt'):
        generate(args.module, config)

    # return

    command = ['nvcc', 'test.cu', '-o', 'test', '-lcublas', '-lcurand']
    print(' '.join(command))
    subprocess.run(command)

    command = ['./test', f'{args.module}']
    print(' '.join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    print('Return code:', result.returncode)
    print('Output:')
    print(result.stdout)

    if not args.save:
        os.remove(f'data/{args.module}.txt')
    if len(os.listdir('data')) == 0:
        os.rmdir('data')


if __name__ == "__main__":
    main()
