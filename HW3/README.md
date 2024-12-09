#### Tensor 类
`/src` 下 `pybind.cu` 注册了 `Tensor` 类的常用函数，包括 `Tensor` 类的等号重载，并实现了 `numpy` 和 `tensor` 的相互转换（深拷贝）

在根目录下 `mnist2tensor.py` 中，使用torch读取MNIST数据集，然后转换为 `numpy` 数组，最后转换为自己的`tensor`

#### Module 
`/src/module` 下实现了七类常用算子及其反传，并已在 `pybind.cu` 中注册，支持在 `python` 侧传入 `tensor` 并在 gpu 上完成计算

#### UnitTest 
在根目录下 `test.py` 实现了部分算子的单元测试，可在 `shape` 中更改参数，如果没有返回信息，说明测试通过