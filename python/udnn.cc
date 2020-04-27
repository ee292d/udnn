#include "../src/udnn.hh"
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "platform.hh"
#include "util.hh"

void init_tensor(py::module &m);
void init_model(py::module &m);

void init_layer(py::module &m) {
  py::class_<LayerBase>(m, "LayerBase")
      .def_property_readonly("in_size",
                             [](LayerBase &l) {
                               auto tuple = py::tuple(4);
                               tuple[0] = l.in_size().x;
                               tuple[1] = l.in_size().y;
                               tuple[2] = l.in_size().c;
                               tuple[3] = l.in_size().k;
                               return tuple;
                             })
      .def_property_readonly("out_size",
                             [](LayerBase &l) {
                               auto tuple = py::tuple(4);
                               tuple[0] = l.out_size().y;
                               tuple[1] = l.out_size().x;
                               tuple[2] = l.out_size().c;
                               tuple[3] = l.out_size().k;
                               return tuple;
                             })
      .def_readwrite("name", &LayerBase::name);

  setup_layer<FlattenLayer, const TensorSize &>(m, "FlattenLayer");
  setup_layer<DenseLayer, const TensorSize &, uint32_t>(m, "DenseLayer");
  setup_layer<Conv2DLayer, const TensorSize &, uint32_t, uint32_t>(
      m, "Conv2DLayer");
}


PYBIND11_MODULE(_udnn, m) {
  init_tensor(m);
  init_layer(m);
  init_model(m);
}
