#include "contract.hpp"
#include "read.hpp"
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
PYBIND11_MODULE(sparse_opcnt, m) {
  m.doc() = "pybind11 example plugin"; // Optional module docstring
  pybind11::class_<Tensor>(m, "Tensor")
      .def(pybind11::init<std::string, bool>())
      .def("write", &Tensor::write)
      .def("count_ops", &Tensor::count_ops)
      .def("output_shape", &Tensor::output_shape);
  pybind11::class_<CoOrdinate>(m, "CoOrdinate")
      .def(pybind11::init<std::vector<int>>());

}
