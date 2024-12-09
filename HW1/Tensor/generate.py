import argparse
import subprocess
import os

import torch
from torch import nn
from torch.nn import functional as F

def to_str(x: torch.Tensor):
    x_str = x.detach().numpy().astype(str)
    x_str = ' '.join(x_str.flatten()) + '\n'
    return x_str

def main():
    parser = argparse.ArgumentParser(
        description="Generate random tensors with a specified shape and write it to a file.")
    parser.set_defaults(delete=False, regenerate=False)
    parser.add_argument('-s', '--shape', dest='shape', nargs='+', type=int, required=True,
                        help="The shape of the array")
    parser.add_argument('-t', '--task', dest='task', type=str, choices=['relu', 'sigmoid'], required=True)
    parser.add_argument('-d', '--delete', dest='delete', action='store_true',
                        help="Delete test data in the end")
    parser.add_argument('-r', '--regenerate', dest='regenerate', action='store_true',
                        help="Regenerate test data")
    args = parser.parse_args()

    if args.regenerate or not os.path.exists('test_data.txt'):
        input = torch.randn(args.shape, requires_grad=True)
        if args.task == 'relu':
            output = F.relu(input)
        else:
            output = F.sigmoid(input)
        out_grad = torch.randn(args.shape, requires_grad=True)
        output.backward(out_grad)
        in_grad = input.grad
        # print(input)
        # print(output)
        # print(in_grad)
        # print(out_grad)
        
        with open('test_data.txt', 'w') as f:
            f.write(to_str(input))
            f.write(to_str(output))
            f.write(to_str(in_grad))
            f.write(to_str(out_grad))
    
    # command = ['tensor_debug'] + [args.task] + list(map(str, args.shape))
    # print(command)
    # command = ' '.join(command)
    command = ['./tensor_debug', args.task] + list(map(str, args.shape))
    result = subprocess.run(command, capture_output=True, text=True)
    print('Return code:', result.returncode)
    print('Output:', result.stdout)
    # print('Error:', result.stderr)

    if args.delete:
        os.remove('test_data.txt')



if __name__ == "__main__":
    main()
