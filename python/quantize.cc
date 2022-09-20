#include "../src/quantize.hh"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void add_quantize(py::module &m) {
  m.def(
       "quantize",
       [](float value, int32_t quantize_bias, int32_t quantize_scale,
          const std::string &dtype) -> int32_t {
         if (dtype != "int8" && dtype != "int16") {
           throw std::runtime_error("Invalid dtype " + dtype);
         }
         if (dtype == "int8") {
           return quantize<int8_t>(value, static_cast<int8_t>(quantize_bias),
                                   static_cast<int8_t>(quantize_scale));
         } else {
           return quantize<int16_t>(value, static_cast<int16_t>(quantize_bias),
                                    static_cast<int16_t>(quantize_scale));
         }
       },
       py::arg("value"), py::arg("quantize_bias"), py::arg("quantize_scale"),
       py::arg("dtype") = "int8")
      .def(
          "quantize_int8",
          [](const Tensor<float> &tensor, int32_t quantize_bias,
             int32_t quantize_scale) {
            return quantize<int8_t, float>(tensor,
                                           static_cast<int8_t>(quantize_bias),
                                           static_cast<int8_t>(quantize_scale));
          },
          py::arg("value"), py::arg("quantize_bias"), py::arg("quantize_scale"))
      .def(
          "quantize_int16",
          [](const Tensor<float> &tensor, int32_t quantize_bias,
             int32_t quantize_scale) {
            return quantize<int16_t, float>(
                tensor, static_cast<int16_t>(quantize_bias),
                static_cast<int16_t>(quantize_scale));
          },
          py::arg("value"), py::arg("quantize_bias"), py::arg("quantize_scale"))
      .def(
          "unquantize",
          [](int32_t value, int32_t quantize_bias, int32_t quantize_scale,
             const std::string &dtype) {
            if (dtype != "int8" && dtype != "int16") {
              throw std::runtime_error("Invalid dtype " + dtype);
            }
            if (dtype == "int8") {
              return unquantize<int8_t>(static_cast<int8_t>(value),
                                        static_cast<int8_t>(quantize_bias),
                                        static_cast<int8_t>(quantize_scale));
            } else {
              return unquantize<int16_t>(static_cast<int16_t>(value),
                                         static_cast<int16_t>(quantize_bias),
                                         static_cast<int16_t>(quantize_scale));
            }
          },
          py::arg("value"), py::arg("quantize_bias"), py::arg("quantize_scale"),
          py::arg("dtype") = "int8")
      .def(
          "unquantize_int8",
          [](const Tensor<int8_t> &tensor, int32_t quantize_bias,
             int32_t quantize_scale) {
            return unquantize<int8_t>(tensor,
                                      static_cast<int8_t>(quantize_bias),
                                      static_cast<int8_t>(quantize_scale));
          },
          py::arg("value"), py::arg("quantize_bias"), py::arg("quantize_scale"))
      .def(
          "unquantize_int16",
          [](const Tensor<int16_t> &tensor, int32_t quantize_bias,
             int32_t quantize_scale) {
            return unquantize<int16_t>(tensor,
                                       static_cast<int16_t>(quantize_bias),
                                       static_cast<int16_t>(quantize_scale));
          },
          py::arg("value"), py::arg("quantize_bias"),
          py::arg("quantize_scale"));

  m.def("fix2float_int8", fix2float<int8_t>);
  m.def("fix2float_int16", fix2float<int16_t>);
  m.def("float2fix_int8", float2fix<int8_t>);
  m.def("float2fix_int16", float2fix<int16_t>);
}

void add_op(py::module &m) {
  m.def(
      "quantize_add",
      [](int32_t a, int32_t b, int32_t quantize_bias, int32_t quantize_scale,
         const std::string &dtype) -> int32_t {
        if (dtype != "int8" && dtype != "int16") {
          throw std::runtime_error("Invalid dtype " + dtype);
        }
        if (dtype == "int8") {
          return quantize_add<int8_t>(static_cast<int8_t>(a),
                                      static_cast<int8_t>(b),
                                      static_cast<int8_t>(quantize_bias),
                                      static_cast<int8_t>(quantize_scale));
        } else {
          return quantize_add<int16_t>(static_cast<int16_t>(a),
                                       static_cast<int16_t>(b),
                                       static_cast<int16_t>(quantize_bias),
                                       static_cast<int16_t>(quantize_scale));
        }
      },
      py::arg("a"), py::arg("b"), py::arg("quantize_bias"),
      py::arg("quantize_scale"), py::arg("dtype") = "int8");

  m.def(
      "quantize_mult",
      [](int32_t a, int32_t b, int32_t quantize_bias, int32_t quantize_scale,
         const std::string &dtype) -> int32_t {
        if (dtype != "int8" && dtype != "int16") {
          throw std::runtime_error("Invalid dtype " + dtype);
        }
        if (dtype == "int8") {
          return quantize_mult<int8_t>(static_cast<int8_t>(a),
                                       static_cast<int8_t>(b),
                                       static_cast<int8_t>(quantize_bias),
                                       static_cast<int8_t>(quantize_scale));
        } else {
          return quantize_mult<int16_t>(static_cast<int16_t>(a),
                                        static_cast<int16_t>(b),
                                        static_cast<int16_t>(quantize_bias),
                                        static_cast<int16_t>(quantize_scale));
        }
      },
      py::arg("a"), py::arg("b"), py::arg("quantize_bias"),
      py::arg("quantize_scale"), py::arg("dtype") = "int8");
}

void init_quantize(py::module &m) {
  add_quantize(m);
  add_op(m);
}