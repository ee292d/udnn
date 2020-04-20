#ifndef UDNN_LAYER_HH
#define UDNN_LAYER_HH

#include "tensor.hh"
#include <cmath>
#include <memory>
#include <stdexcept>
#include <type_traits>

enum class DType { Int8, Int16, Int32, Int64, Float, Double };

class LayerBase {
public:
  virtual TensorSize in_size() const = 0;
  virtual TensorSize out_size() const = 0;
  virtual DType in_type() const = 0;
  virtual DType out_type() const = 0;

  virtual const TensorBase *out_base() const = 0;

  std::string name;

  inline LayerBase *connect(LayerBase *next) {
    if (!next) {
      next_ = nullptr;
      return nullptr;
    }
    if (next->in_size() != out_size())
      throw std::invalid_argument(
          "Tensor dimension mismatch: " + next->in_size().str() + " -> " +
          out_size().str());
    if (next->in_type() != out_type())
      throw std::invalid_argument("Tensor type mismatch");
    next_ = next;
    return next;
  }
  inline LayerBase *next() const { return next_; }
  virtual void forward(void *data) = 0;

  ~LayerBase() = default;

private:
  LayerBase *next_ = nullptr;
};

template <typename T> class Layer : public LayerBase {
public:
  inline Layer(const TensorSize &in_size, const TensorSize &out_size)
      : in_size_(in_size), out_(out_size) {}

  inline Layer() = default;

  inline TensorSize in_size() const override { return in_size_; }
  inline TensorSize out_size() const override { return out_.size; }
  inline DType in_type() const override { return get_type<T>(); }
  inline DType out_type() const override { return get_type<T>(); }

  inline virtual const Tensor<T> &out() const { return out_; }
  inline const TensorBase *out_base() const override { return &out_; }

  virtual void forward(const Tensor<T> &input) = 0;

  // noop for layers that doesn't have weights
  inline virtual void load_weights(const Tensor<T> &) {}
  // first one is weights, second one is bias
  inline virtual void load_bias(const Tensor<T> &) {}
  inline virtual bool has_weights() const { return false; }
  inline virtual bool has_bias() const { return false; }
  inline virtual TensorSize weights_size() const { return {0, 0, 0, 0}; }
  inline virtual const Tensor<T> *get_weights() const { return nullptr; }
  inline virtual const Tensor<T> *get_bias() const { return nullptr; }
  inline virtual TensorSize weight_size() const { return {0, 0, 0, 0}; }
  inline virtual TensorSize bias_size() const { return {0, 0, 0, 0}; }

  inline void forward(void *data) override {
    // do not copy the data
    auto t = Tensor<T>(data, in_size(), TensorSize::default_stride(in_size()),
                       false);
    forward(t);
  }

protected:
  TensorSize in_size_;
  Tensor<T> out_;

private:
  template <typename V> inline static DType get_type() {
    static_assert(std::is_fundamental<V>(),
                  "Template type has to be numeric types");
    if (std::is_same<V, int8_t>())
      return DType::Int8;
    else if (std::is_same<V, int16_t>())
      return DType::Int16;
    else if (std::is_same<V, int32_t>())
      return DType::Int32;
    else if (std::is_same<V, int64_t>())
      return DType::Int64;
    else if (std::is_same<V, float>())
      return DType::Float;
    else if (std::is_same<V, double>())
      return DType::Double;
    else
      throw std::runtime_error("Unable to determine types");
  }
};

template <typename T> class Conv2DLayer : public Layer<T> {
public:
  uint32_t filter_size;
  uint32_t num_filters;

  inline Conv2DLayer(const TensorSize &in_size, uint32_t filter_size,
                     uint32_t num_filters) {}

  inline void set_weight(uint32_t y, uint32_t x, uint32_t c, uint32_t k,
                         T value) {}

  inline void load_weights(const Tensor<T> &weight) override {
    weights_.load(weight);
  }

  inline TensorSize weights_size() const { return weights_.size; }

  inline void forward(const Tensor<T> &in) override {}

  inline TensorSize weight_size() const override { return weights_.size; }
  inline TensorSize bias_size() const override { return bias_.size; }
  inline virtual const Tensor<T> *get_weights() const { return &weights_; }
  inline virtual const Tensor<T> *get_bias() const { return &bias_; }

  inline bool has_bias() const override { return true; }
  inline bool has_weights() const override { return true; }
  inline void load_bias(const Tensor<T> &bias) override { bias_.load(bias); }

private:
  Tensor<T> weights_;
  Tensor<T> bias_;
};

template <typename T> class MaxPoolingLayer : public Layer<T> {
public:
  inline explicit MaxPoolingLayer(const TensorSize &in_size,
                                  uint32_t pool_size) {}

  inline void forward(const Tensor<T> &in) override {}

private:
  uint32_t pool_size_;
};

template <typename T> class FlattenLayer : public Layer<T> {
public:
  inline explicit FlattenLayer(const TensorSize &in_size) {}

  inline void forward(const Tensor<T> &in) override {}
};

template <typename T> class DenseLayer : public Layer<T> {
public:
  inline DenseLayer(const TensorSize &in_size, uint32_t out_size) {}

  inline void set_weight(uint32_t y, uint32_t x, uint32_t c, T value) {}

  inline void forward(const Tensor<T> &in) override {}

  inline bool has_bias() const override { return true; }
  inline bool has_weights() const override { return true; }

  inline void load_weights(const Tensor<T> &weight) override {
    weights_.load(weight);
  }
  inline TensorSize weight_size() const override { return weights_.size; }
  inline TensorSize bias_size() const override { return bias_.size; }

  inline void load_bias(const Tensor<T> &bias) override { bias_.load(bias); }

  inline TensorSize weights_size() const { return weights_.size; }
  inline virtual const Tensor<T> *get_weights() const { return &weights_; }
  inline virtual const Tensor<T> *get_bias() const { return &bias_; }

protected:
  Tensor<T> weights_;
  Tensor<T> bias_;
};

template <typename T> class ActivationLayer : public Layer<T> {
public:
  inline explicit ActivationLayer(const TensorSize &size)
      : Layer<T>(size, size) {}

  inline void forward(const Tensor<T> &in) {}

protected:
  inline virtual T activate_function(T value) { return value; }
};

template <typename T> class ReLuActivationLayer : public ActivationLayer<T> {
public:
  inline explicit ReLuActivationLayer(const TensorSize &size)
      : ActivationLayer<T>(size) {}

protected:
  inline T activate_function(T value) override { return 0; }
};

template <typename T> class SigmoidActivationLayer : public ActivationLayer<T> {
public:
  inline explicit SigmoidActivationLayer(const TensorSize &size)
      : ActivationLayer<T>(size) {}

protected:
  inline T activate_function(T value) override { return 0; }
};

#endif // UDNN_LAYER_HH
