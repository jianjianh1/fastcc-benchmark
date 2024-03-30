#include "contract.hpp"
#include "read.hpp"
#include "types.hpp"
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
PYBIND11_MODULE(sparse_opcnt, m) {
  m.doc() = "pybind11 example plugin"; // Optional module docstring
  pybind11::class_<Tensor<double>>(m, "TensorScalar")
      .def(pybind11::init<std::string, bool>())
      .def("write", &Tensor<double>::write)
      .def("count_ops_ss", &Tensor<double>::count_ops<double>)
      .def("count_ops_sv", &Tensor<double>::count_ops<densevec>)
      .def("count_ops_sm", &Tensor<double>::count_ops<densemat>)
      .def("output_shape", &Tensor<double>::output_shape)
      .def("contract", &Tensor<double>::contract);
  pybind11::class_<Tensor<densevec>>(m, "TensorDenseVec")
      .def(pybind11::init<std::string, bool>())
      .def("write", &Tensor<densevec>::write)
      .def("count_ops_vv", &Tensor<densevec>::count_ops<densevec>)
      .def("count_ops_vm", &Tensor<densevec>::count_ops<densemat>)
      .def("count_ops_vs", &Tensor<densevec>::count_ops<double>)
      .def("output_shape", &Tensor<densevec>::output_shape)
      .def("contract", &Tensor<densevec>::contract);
  pybind11::class_<Tensor<densemat>>(m, "TensorDenseMat")
      .def(pybind11::init<std::string, bool>())
      .def("write", &Tensor<densemat>::write)
      .def("count_ops_mv", &Tensor<densemat>::count_ops<densevec>)
      .def("count_ops_ms", &Tensor<densemat>::count_ops<double>)
      .def("count_ops_mm", &Tensor<densemat>::count_ops<densemat>) // This is really stupid, is there another way?
      .def("output_shape", &Tensor<densemat>::output_shape)
      .def("contract", &Tensor<densemat>::contract);
  pybind11::class_<CoOrdinate>(m, "CoOrdinate")
      .def(pybind11::init<std::vector<int>>());
}
