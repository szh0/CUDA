# HW2

- [HW2](#HW2)
  - [Introduction](#Introduction)
  - [Test](#Test)

## Introduction

[module.cuh](module.cuh)
  - core forward and backward functions of modules, as well as some helper functions and kernel functions

[test.cu](test.cu)
  - read test data and test the module

[generate.py](generate.py)
  - generate data randomly and write them to a file
  - compile and run [test.cu](test.cu)

[config.json](config.json)
  - specify the shape of the test data

Parameters of the forward and backward functions contains pointers of input and output data `float*` and their shapes `int`, e.g.,
```
void backward_fc(float *input, float *output, float *weight, float *bias,
                 int batchsize, int in_features, int out_features,
                 float *grad_output, float *grad_input,
                 float *grad_weights, float *grad_bias);
```

`gemm_gpu` function repackages `cublasSgemm` to suit for row-major array layout data, since `cuBLAS` code uses column-major array layout.

All the data in gpu are stored in row-major array layout, so that it is easier to move them between cpu and gpu.

The fully connected layer and the convolution layer include the bias item.

The convolution layer and the max pooling layer can use customized parameters, including `ksize`, `pad`, `stride`.

However, `stride` should be equal to `ksize` in the max pooling layer.

## Test

Run [generate.py](generate.py) to test the modules.

```
usage: generate.py [-h] -m {fc,conv,maxpool,celoss} [-s] [-r]

Generate random data and test modules.

options:
  -h, --help            show this help message and exit
  -m {fc,conv,maxpool,celoss}, --module {fc,conv,maxpool,celoss}
  -s, --save            Save test data
  -r, --regenerate      Regenerate test data
```

To change the shape of the test data, modify [config.json](config.json).

To test `forward_fc` and `backward_fc`, run
```
python generate.py -m fc
```

To test `forward_conv` and `backward_conv`, run
```
python generate.py -m conv
```

To test `forward_maxpool` and `backward_maxpool`, run
```
python generate.py -m maxpool
```

To test `forward_softmax`, `forward_celoss`, and `forward_celoss`, run
```
python generate.py -m celoss
```

