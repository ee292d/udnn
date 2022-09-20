#include "../src/udnn.hh"
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "platform.hh"
#include "util.hh"

void init_model(py::module &m) {
  auto model =
      py::class_<Model>(m, "Model")
          .def(py::init<>())
          .def("add_layer", &Model::add_layer)
          .def("predict", &Model::predict)
          .def("predict_simd", &Model::predict_simd)
          // out is managed by the layer, not by Python. Hence, we let the C++
          // take care of the ownership
          .def("out", &Model::out, py::return_value_policy::reference_internal)
          .def("out_type",
               [](Model &model_) {
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
               })
          .def_readwrite("quantized", &Model::quantized);

  setup_model_convert_out<int8_t>(model);
  setup_model_convert_out<int16_t>(model);
  setup_model_convert_out<int32_t>(model);
  setup_model_convert_out<int64_t>(model);
  setup_model_convert_out<float>(model);
  setup_model_convert_out<double>(model);
}