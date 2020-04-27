#include "../src/udnn.hh"
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "platform.hh"
#include "util.hh"

void init_tensor(py::module &m);

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

void init_model(py::module &m) {
  auto model =
      py::class_<Model>(m, "Model")
          .def(py::init<>())
          .def("add_layer", &Model::add_layer)
          .def("predict", &Model::predict)
          // out is managed by the layer, not by Python. Hence we let the C++
          // take care of the ownership
          .def("out", &Model::out, py::return_value_policy::reference_internal)
          .def("out_type", [](Model &model_) {
            auto out_type = model_.out_type();
            switch (out_type) {
            case DType::Int8:
              return type_to_str<int8_t>();
            case DType::Int16:
              return type_to_str<int16_t>();
            case DType::Int32:
              return type_to_str<int32_t>();
            case DType::Int64:
              return type_to_str<int64_t>();
            case DType::Float:
              return type_to_str<float>();
            case DType::Double:
              return type_to_str<double>();
            default:
              return type_to_str<float>();
            }
          });

  setup_model_convert_out<int8_t>(model);
  setup_model_convert_out<int16_t>(model);
  setup_model_convert_out<int32_t>(model);
  setup_model_convert_out<int64_t>(model);
  setup_model_convert_out<float>(model);
  setup_model_convert_out<double>(model);
}

PYBIND11_MODULE(_udnn, m) {
  init_tensor(m);
  init_layer(m);
  init_model(m);
}
