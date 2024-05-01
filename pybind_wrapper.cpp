#include "contract.hpp"
#include "read.hpp"
#include "task_queue.hpp"
#include "types.hpp"
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
PYBIND11_MODULE(sparse_opcnt, m) {
  m.doc() = "pybind11 example plugin"; // Optional module docstring
  pybind11::class_<Tensor<double>>(m, "TensorScalar")
      .def(pybind11::init<std::string, bool>())
      .def(pybind11::init<int>())
      .def("write", &Tensor<double>::write);
  //.def("count_ops_ss", &Tensor<double>::count_ops<double>)
  //.def("count_ops_sv", &Tensor<double>::count_ops<densevec>)
  //.def("count_ops_sm", &Tensor<double>::count_ops<densemat>)
  //.def("output_shape", &Tensor<double>::output_shape)
  //.def("contract", &Tensor<double>::contract);
  pybind11::class_<Tensor<densevec>>(m, "TensorDenseVec")
      .def(pybind11::init<std::string, bool>())
      .def(pybind11::init<int>())
      .def("write", &Tensor<densevec>::write);
  //.def("count_ops_vv", &Tensor<densevec>::count_ops<densevec>)
  //.def("count_ops_vm", &Tensor<densevec>::count_ops<densemat>)
  //.def("count_ops_vs", &Tensor<densevec>::count_ops<double>)
  //.def("output_shape", &Tensor<densevec>::output_shape)
  //.def("contract", &Tensor<densevec>::contract);
  pybind11::class_<Tensor<densemat>>(m, "TensorDenseMat")
      .def(pybind11::init<std::string, bool>())
      .def(pybind11::init<int>())
      .def("write", &Tensor<densemat>::write);
  //.def("count_ops_mv", &Tensor<densemat>::count_ops<densevec>)
  //.def("count_ops_ms", &Tensor<densemat>::count_ops<double>)
  //.def("count_ops_mm", &Tensor<densemat>::count_ops<densemat>) // This is
  //really stupid, is there another way? .def("output_shape",
  //&Tensor<densemat>::output_shape) .def("contract",
  //&Tensor<densemat>::contract);
  pybind11::class_<SymbolicTensor>(m, "SymbolicTensor")
      .def(pybind11::init<Tensor<double> &>())
      .def(pybind11::init<Tensor<densemat> &>())
      .def(pybind11::init<Tensor<densevec> &>())
      .def("count_ops", &SymbolicTensor::count_ops)
      .def("contract", &SymbolicTensor::contract);
  pybind11::class_<CoOrdinate>(m, "CoOrdinate")
      .def(pybind11::init<std::vector<int>>());
  pybind11::class_<TaskQueue>(m, "TaskQueue")
      .def(pybind11::init<>())
      .def("make_task_ss", &TaskQueue::makeTask<double, double, double>)
      .def("make_task_sv", &TaskQueue::makeTask<densevec, double, densevec>)
      .def("make_task_vs", &TaskQueue::makeTask<densevec, densevec, double>)
      .def("make_task_sm", &TaskQueue::makeTask<densemat, double, densemat>)
      .def("make_task_ms", &TaskQueue::makeTask<densemat, densemat, double>)
      .def("make_task_svv", &TaskQueue::makeTask<double, densevec, densevec>)
      .def("make_task_mvv", &TaskQueue::makeTask<densemat, densevec, densevec>)
      .def("make_task_mm", &TaskQueue::makeTask<densemat, densemat, densemat>)
      .def("make_task_mv", &TaskQueue::makeTask<densevec, densemat, densevec>)
      .def("make_task_vm", &TaskQueue::makeTask<densevec, densevec, densemat>)
      .def("update_doubles", &TaskQueue::updateDoubles)
      .def("write_doubles", &TaskQueue::writeDoubles)
      .def("get_t2", &TaskQueue::getDoubles)
      .def("loop_until", &TaskQueue::loopUntil);
  pybind11::class_<tf::Task>(m, "Task").def("precede",
                                            &tf::Task::precede<tf::Task &>);
}
