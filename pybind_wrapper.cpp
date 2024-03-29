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
      .def("count_ops", &Tensor<double>::count_ops)
      .def("output_shape", &Tensor<double>::output_shape)
      .def("contract", &Tensor<double>::contract);
  pybind11::class_<Tensor<densevec>>(m, "TensorDenseVec")
      .def(pybind11::init<std::string, bool>())
      .def("write", &Tensor<densevec>::write)
      .def("count_ops", &Tensor<densevec>::count_ops)
      .def("output_shape", &Tensor<densevec>::output_shape)
      .def("contract", &Tensor<densevec>::contract);
  pybind11::class_<Tensor<densemat>>(m, "TensorDenseMat")
      .def(pybind11::init<std::string, bool>())
      .def("write", &Tensor<densemat>::write)
      .def("count_ops", &Tensor<densemat>::count_ops)
      .def("output_shape", &Tensor<densemat>::output_shape)
      .def("contract", &Tensor<densemat>::contract);
  pybind11::class_<CoOrdinate>(m, "CoOrdinate")
      .def(pybind11::init<std::vector<int>>());
}
