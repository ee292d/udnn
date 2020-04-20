#include "../src/udnn.hh"
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "platform.hh"

namespace py = pybind11;

template <typename T> constexpr const char *type_to_str() {
  if (std::is_same<T, int8_t>()) {
    return "Int8";
  } else if (std::is_same<T, int16_t>()) {
    return "Int16";
  } else if (std::is_same<T, int32_t>()) {
    return "Int32";
  } else if (std::is_same<T, int64_t>()) {
    return "Int64";
  } else if (std::is_same<T, float>()) {
    return "Float";
  } else if (std::is_same<T, double>()) {
    return "Double";
  } else {
    throw std::invalid_argument("Unable to convert type");
  }
}

template <typename T>
Tensor<T> *create_tensor_from_buffer(const py::buffer &b, bool copy = true) {
  auto info = b.request();
  /* Some sanity checks ... */
  // notice the bug in python data types
  // where long (int64_t) is reported as l, which could be 32-bit one
  // on 32-bit machines, which could be rare
  auto fmt = info.format;
  if (ENV64 && fmt == "l")
    fmt = "q"; // NOLINT

  if (fmt != py::format_descriptor<T>::format()) {
    std::string type_name = py::format_descriptor<T>::format();
    throw std::runtime_error("Incompatible format: expected a " + type_name +
                             " array! Got " + info.format);
  }

  if (info.ndim != 3 && info.ndim != 4)
    throw std::runtime_error("Incompatible buffer dimension!");
  auto y = static_cast<uint32_t>(info.shape[0]);
  auto x = static_cast<uint32_t>(info.shape[1]);
  auto c = static_cast<uint32_t>(info.shape[2]);
  uint32_t k = 1;
  if (info.ndim == 4)
    k = info.shape[3];
  auto x_stride = static_cast<uint32_t>(info.strides[1]);
  auto y_stride = static_cast<uint32_t>(info.strides[0]);
  auto c_stride = static_cast<uint32_t>(info.strides[2]);
  auto k_stride = 1;
  if (info.ndim == 4)
    k_stride = static_cast<uint32_t>(info.strides[3]);

  auto size = static_cast<uint32_t>(sizeof(T));
  return new Tensor<T>(
      info.ptr, TensorSize{y, x, c, k},
      {y_stride / size, x_stride / size, c_stride / size, k_stride / size},
      copy);
}

template <typename T> py::class_<Tensor<T>> setup_tensor(py::module &m) {
  auto suffix = type_to_str<T>();
  auto name = std::string("Tensor").append(suffix);
  return py::class_<Tensor<T>, TensorBase>(m, name.c_str(),
                                           py::buffer_protocol())
      .def_buffer([](Tensor<T> &t) -> py::buffer_info {
        auto size = sizeof(T);
        auto stride = t.stride();
        return py::buffer_info(
            t.ptr(),                                  /* Pointer to buffer */
            sizeof(T),                                /* Size of one scalar */
            py::format_descriptor<T>::format(),       /* Format descriptor */
            4,                                        /* Number of dimensions */
            {t.size.y, t.size.x, t.size.c, t.size.k}, /* Buffer dimensions */
            {stride.y * size, stride.x * size, stride.c * size, stride.k * size}
            /* Strides (in bytes) for each index */
        );
      })
      .def(py::init([suffix](const py::buffer &b) {
        return create_tensor_from_buffer<T>(b);
      }))
      .def(py::init([](const py::tuple &shape) {
        if (shape.size() != 3 && shape.size() != 4)
          throw std::runtime_error("Expect a size 3 or 4 tuple");
        auto const &y = static_cast<uint32_t>(py::int_(shape[0]));
        auto const &x = static_cast<uint32_t>(py::int_(shape[1]));
        auto const &c = static_cast<uint32_t>(py::int_(shape[2]));
        uint32_t k = 1;
        if (shape.size() == 4)
          k = static_cast<uint32_t>(py::int_(shape[3]));
        return new Tensor<T>(y, x, c, k);
      }))
      .def("__getitem__",
           [](Tensor<T> &t,
              const std::tuple<uint32_t, uint32_t, uint32_t, uint32_t> &index)
               -> T {
             auto const &y = std::get<0>(index);
             auto const &x = std::get<1>(index);
             auto const &c = std::get<2>(index);
             auto const &k = std::get<2>(index);
             return t(y, x, c, k);
           })
      .def("__getitem__",
           [](Tensor<T> &t,
              const std::tuple<uint32_t, uint32_t, uint32_t> &index) -> T {
             if (t.size.k != 1)
               throw std::invalid_argument(
                   "4th dimension is not 1. need 4 indices");
             auto const &y = std::get<0>(index);
             auto const &x = std::get<1>(index);
             auto const &c = std::get<2>(index);
             return t(y, x, c);
           })
      .def("dump", py::overload_cast<const std::string &>(&Tensor<T>::dump))
      .def("load", &Tensor<T>::load_from_file);
}

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

template <template <typename> class C, typename T, typename... ctor>
void setup_layer_t(py::module &m, std::string base_name) {
  auto suffix = type_to_str<T>();
  auto name = base_name.append(suffix).c_str();
  py::class_<C<T>, LayerBase>(m, name)
      .def(py::init<ctor...>())
      .def("forward", &C<T>::forward)
      .def("forward",
           [](C<T> &layer, const py::buffer &b) {
             auto t = create_tensor_from_buffer<T>(b);
             layer.forward(*t);
             // free it
             delete t;
           })
      .def_property_readonly("out", &C<T>::out)
      .def("load_weights",
           py::overload_cast<const Tensor<T> &>(&C<T>::load_weights))
      .def("load_weights",
           [](C<T> &layer, const py::buffer &b) {
             auto tensor = create_tensor_from_buffer<T>(b);
             // load weights
             layer.load_weights(*tensor);
             // need to delete it to avoid memory leak
             delete tensor;
           })
      // return reference since the C++ object is managing the memory
      .def_property_readonly("weights", &C<T>::get_weights,
                             py::return_value_policy::reference)
      .def_property_readonly("bias", &C<T>::get_bias,
                             py::return_value_policy::reference)
      .def_property_readonly("has_bias", &C<T>::has_bias)
      .def_property_readonly("has_weights", &C<T>::has_weights)
      .def("load_bias", &C<T>::load_bias)
      .def("load_bias",
           [](C<T> &layer, const py::buffer &b) {
             auto tensor = create_tensor_from_buffer<T>(b);
             // load bias
             layer.load_bias(*tensor);
             // need to delete it to avoid memory leak
             delete tensor;
           })
      .def_property_readonly("weights_size", &C<T>::weights_size)
      .def_property_readonly("bias_size", &C<T>::bias_size);
}

template <template <typename> class C, typename... ctor>
void setup_layer(py::module &m, std::string base_name) {
  setup_layer_t<C, int8_t, ctor...>(m, base_name);
  setup_layer_t<C, int16_t, ctor...>(m, base_name);
  setup_layer_t<C, int32_t, ctor...>(m, base_name);
  setup_layer_t<C, int64_t, ctor...>(m, base_name);
  setup_layer_t<C, float, ctor...>(m, base_name);
  setup_layer_t<C, double, ctor...>(m, base_name);
}

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

template <typename T> void setup_model_convert_out(py::class_<Model> &model) {
  std::string name = "out_as_" + std::string(type_to_str<T>());
  model.def(name.c_str(), [](Model &m) -> Tensor<T> * {
    auto out = m.out();
    if (out->element_size() != sizeof(T))
      throw std::invalid_argument("Unable to convert tensor: type mismatch");
    // we copy the data here since the memory is managed by the layer,
    // which may get cleared while the tensor is still valid
    return new Tensor<T>(out->ptr(), out->size,
                         TensorSize::default_stride(out->size), true);
  });
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
