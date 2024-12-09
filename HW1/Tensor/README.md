## 测试

首先用以下命令编译[tensor_debug.cu](tensor_debug.cu)。

`nvcc tensor_debug.cu -o tensor_debug`

用以下命令对代码进行测试。

`python generate.py [-h] -s SHAPE [SHAPE ...] -t {relu,sigmoid} [-d] [-r]`

通过`-s`参数指定Tensor形状，通过`-t`参数指定测试的激活函数。
可以通过`-d`参数将生成的数据在测试后删除。
可以通过`-r`参数使存在数据文件时仍重新生成数据。

示例：

`python generate.py -s 2 3 4 -t relu -d`将随机生成(2, 3, 4)形状的Tensor `input`，`out_grad`，通过pytorch计算`input`经过ReLU的结果`output`，和下游梯度`in_grad`，然后将它们的数据写入[test_data.txt](test_data.txt)，运行[tensor_debug.exe](tensor_debug.exe)并打印输出，最后删除[test_data.txt](test_data.txt)。

[tensor_debug.cu](tensor_debug.cu)会读取[test_data.txt](test_data.txt)中的数据，分别在cpu，gpu上测试实现的激活函数的正向转播和反向传播函数。此外还会对Tensor类的成员函数作不完全的检查。
