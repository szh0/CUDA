#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tensor.h"
#include "utils.cuh"
#include "tensor_op.cuh"
#include "module/sigmoid.cuh"
#include "module/relu.cuh"
#include "module/fc.cuh"
#include "module/maxpool.cuh"
#include "module/conv.cuh"
#include "module/softmax.cuh"
#include "module/cross_entropy.cuh"

namespace py = pybind11;

PYBIND11_MODULE(mytensor, m) {
    py::class_<Tensor>(m,"Tensor")
    .def(py::init<const std::vector<int> &, const std::string &>())
    .def("size", &Tensor::size)
    .def("set_data", &Tensor::set_data)
    .def("cpu", &Tensor::cpu)
    .def("gpu", &Tensor::gpu)
    .def("get_data", &Tensor::get_data)
    .def("print_data", &Tensor::print_data)
    .def("__eq__", &Tensor::operator==);

    //device:gpu
    m.def("tensor_init", &tensor_init, "A function that initializes a tensor on the GPU");
    m.def("zeros_init", &zero_init, "A function that initializes a tensor with zeros on the GPU");

    m.def("sigmoid_forward", &sigmoid_forward_wrapper, "input, output");
    m.def("sigmoid_backward", &sigmoid_backward_wrapper, "output, grad_output, grad_input");
    
    m.def("relu_forward", &relu_forward_wrapper, "input, output");
    m.def("relu_backward", &relu_backward_wrapper, "output, grad_output, grad_input");

    m.def("fc_forward", &fc_forward_wrapper, "input, output, weight, bias");
    m.def("fc_backward", &fc_backward_wrapper, "input, output, weight, bais, grad_input, grad_output, grad_weight, grad_bias");

    m.def("maxpool_forward", &maxpool_forward_wrapper, "input, output, mask, kernel_size, stride, padding");
    m.def("maxpool_backward", &maxpool_backward_wrapper, "input, output, mask, grad_input, grad_output, kernel_size, stride, padding");

    m.def("conv_forward", &conv_forward_wrapper, "input, output, kernel, bias, kernel_size, stride, padding");
    m.def("conv_backward", &conv_backward_wrapper, "input, output, kernel, bias, grad_input, grad_output, grad_kernel, grad_bias, kernel_size, stride, padding");

    m.def("softmax_forward", &softmax_forward_wrapper, "input, output");

    m.def("cross_entropy_forward", &cross_entropy_forward_wrapper, "input, target, output");
    m.def("cross_entropy_backward", &cross_entropy_backward_wrapper, "with softmax/input, grad_prob, target, loss, grad_input");

    m.def("np2tensor", &numpy_to_tensor, "A function that converts a numpy array to a Tensor");
    m.def("tensor2np", &tensor_to_numpy, "A function that converts a Tensor to a numpy array");
}


