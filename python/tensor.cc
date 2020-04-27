#include "../src/udnn.hh"
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "platform.hh"
#include "util.hh"


void init_tensor(py::module &m) {
  py::class_<TensorBase>(m, "TensorBase")
      .def_readonly("owned", &TensorBase::owned)
      .def_property_readonly("shape", [](TensorBase &t) {
        auto tuple = py::tuple(4);
        tuple[0] = t.size.y;
        tuple[1] = t.size.x;
        tuple[2] = t.size.c;
        tuple[3] = t.size.k;
        return tuple;
      });
  setup_tensor<int8_t>(m);
  setup_tensor<int16_t>(m);
  setup_tensor<int32_t>(m);
  setup_tensor<int64_t>(m);
  setup_tensor<float>(m);
  setup_tensor<double>(m);

  // tensor type
  py::class_<TensorSize>(m, "TensorSize")
      .def(py::init([](const py::tuple &shape) {
        if (shape.size() != 3 && shape.size() != 4)
          throw std::runtime_error("Expect a size 3 or 4 tuple");
        auto const &x = static_cast<uint32_t>(py::int_(shape[0]));
        auto const &y = static_cast<uint32_t>(py::int_(shape[1]));
        auto const &c = static_cast<uint32_t>(py::int_(shape[2]));
        uint32_t k = 1;
        if (shape.size() == 4)
          k = static_cast<uint32_t>(py::int_(shape[3]));
        return new TensorSize{x, y, c, k};
      }))
      .def("tuple",
           [](TensorSize &t) { return py::make_tuple(t.y, t.x, t.c, t.k); });
}