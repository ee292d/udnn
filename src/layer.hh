#ifndef UDNN_LAYER_HH
#define UDNN_LAYER_HH

#include "quantize.hh"
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
  virtual void forward_simd(void *data) = 0;
  virtual void forward_quantized(void *data) = 0;

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
  virtual void forward_simd(const Tensor<T> &input) = 0;
  virtual void forward_quantized(const Tensor<T> &input) = 0;

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

  inline void forward_simd(void *data) override {
    // do not copy the data
    auto t = Tensor<T>(data, in_size(), TensorSize::default_stride(in_size()),
                       false);
    forward_simd(t);
  }

  inline void forward_quantized(void *data) override {
    // do not copy the data
    auto t = Tensor<T>(data, in_size(), TensorSize::default_stride(in_size()),
                       false);
    forward_quantized(t);
  }

  T quantization_bias = 0;
  T quantization_scale = 1;

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
                     uint32_t num_filters)
      : Layer<T>(in_size, TensorSize{in_size.y - filter_size + 1,
                                     in_size.x - filter_size + 1, num_filters,
                                     in_size.k}), /* no padding */
        filter_size(filter_size), num_filters(num_filters),
        weights_(TensorSize{filter_size, filter_size, in_size.c,
                            num_filters}), /* set weights dimension */
        bias_(TensorSize{1, 1, num_filters, 1}) /*set bias dimension */ {}

  inline void set_weight(uint32_t y, uint32_t x, uint32_t c, uint32_t k,
                         T value) {
    weights_(y, x, c, k) = value;
  }

  inline void load_weights(const Tensor<T> &weight) override {
    weights_.load(weight);
  }

  inline TensorSize weights_size() const { return weights_.size; }

  inline void forward(const Tensor<T> &in) override {
    // TODO:
    //    IMPLEMENT THIS!
  }

  inline void forward_simd(const Tensor<T> &in) override {
    // TODO:
    //    IMPLEMENT THIS!
  }

  inline void forward_quantized(const Tensor<T> &in) override {
    // TODO:
    //    IMPLEMENT THIS!
  }

  inline TensorSize weight_size() const override { return weights_.size; }
  inline TensorSize bias_size() const override { return bias_.size; }
  inline virtual const Tensor<T> *get_weights() const { return &weights_; }
  inline virtual const Tensor<T> *get_bias() const { return &bias_; }

  inline bool has_bias() const override { return true; }
  inline bool has_weights() const override { return true; }
  inline void load_bias(const Tensor<T> &bias) override { bias_.load(bias); }

  // quantization
  template <typename K>
  Conv2DLayer<K> quantize(K quantization_bias, K quantization_scale) const {
    Conv2DLayer<K> new_layer(this->in_size_, filter_size, num_filters);
    new_layer.quantization_bias = quantization_bias;
    new_layer.quantization_scale = quantization_scale;
    auto w = ::quantize<K>(weights_, quantization_bias, quantization_scale);
    new_layer.load_weights(w);
    auto b = ::quantize<K>(bias_, quantization_bias, quantization_scale);
    new_layer.load_bias(b);
    return new_layer;
  }

private:
  Tensor<T> weights_;
  Tensor<T> bias_;
};

template <typename T> class MaxPoolingLayer : public Layer<T> {
public:
  inline explicit MaxPoolingLayer(const TensorSize &in_size, uint32_t pool_size)
      : Layer<T>(in_size, /* input size */
                 TensorSize{in_size.y / pool_size, in_size.x / pool_size,
                            in_size.c, in_size.k}), /* output size */
        pool_size_(pool_size) {}

  inline void forward(const Tensor<T> &in) override {
    // in our system batch size is always 1
    constexpr auto k = 0;
    // loop over channels
    for (auto c = 0; c < this->in_size_.c; c++) {
      // loop over x dim
      for (auto x = 0; x < this->out_.size.x; x++) {
        // loop over y
        for (auto y = 0; y < this->out_.size.y; y++) {
          // use the first element to initialize
          T max = in(y * this->pool_size_, x * this->pool_size_, c, k);
          // loop over pooling filter x dim
          for (auto i = 0; i < this->pool_size_; i++) {
            // loop over pooling filter y dim
            for (auto j = 0; j < this->pool_size_; j++) {
              // notice the access is always (y, x, c, k)
              T value =
                  in(y * this->pool_size_ + j, x * this->pool_size_ + i, c, k);
              if (value > max) {
                max = value;
              }
            }
          }
          this->out_(y, x, c, k) = max;
        }
      }
    }
  }

  inline void forward_simd(const Tensor<T> &in) override {
    forward(in);
  }

  inline void forward_quantized(const Tensor<T> &in) override {
    forward(in);
  }

  // quantization
  template <typename K>
  MaxPoolingLayer<K> quantize(K quantization_bias, K quantization_scale) const {
    MaxPoolingLayer<K> new_layer(this->in_size_, pool_size_);
    new_layer.quantization_bias = quantization_bias;
    new_layer.quantization_scale = quantization_scale;
    // noop
    return new_layer;
  }

private:
  uint32_t pool_size_;
};

template <typename T> class FlattenLayer : public Layer<T> {
public:
  inline explicit FlattenLayer(const TensorSize &in_size)
      : Layer<T>(in_size, TensorSize{1, in_size.y * in_size.x * in_size.c, 1,
                                     in_size.k}) {}

  inline void forward(const Tensor<T> &in) override {
    // simply loop through the loop and copy data over
    // there is a more efficient solution!
    // hint: think of the tensor memory layout and what is flattening!
    constexpr auto k = 0; /* batch_size = 1 */
    auto count = 0;
    for (auto y = 0; y < this->in_size_.y; y++) {
      for (auto x = 0; x < this->in_size_.x; x++) {
        for (auto c = 0; c < this->in_size_.c; c++) {
          this->out_(0, count++, 0, k) = in(y, x, c, k);
        }
      }
    }
  }

  inline void forward_simd(const Tensor<T> &in) override {
    forward(in);
  }

  inline void forward_quantized(const Tensor<T> &in) override {
    forward(in);
  }

  // quantization
  template <typename K>
  FlattenLayer<K> quantize(K quantization_bias, K quantization_scale) const {
    FlattenLayer<K> new_layer(this->in_size_);
    new_layer.quantization_bias = quantization_bias;
    new_layer.quantization_scale = quantization_scale;
    // noop
    return new_layer;
  }
};

template <typename T> class DenseLayer : public Layer<T> {
public:
  inline DenseLayer(const TensorSize &in_size, uint32_t out_size)
      : Layer<T>(in_size, TensorSize{1, out_size, 1, in_size.k}),
        out_size_(out_size), weights_(TensorSize{in_size.x, out_size, 1, 1}),
        bias_(TensorSize{1, out_size, 1, 1}) {}

  inline void set_weight(uint32_t y, uint32_t x, uint32_t c, uint32_t k,
                         T value) {
    weights_(y, x, c, k) = value;
  }

  inline void forward(const Tensor<T> &in) override {
    // TODO:
    //    IMPLEMENT THIS!
  }

  inline void forward_simd(const Tensor<T> &in) override {
    // TODO:
    //    IMPLEMENT THIS!
  }

  inline void forward_quantized(const Tensor<T> &in) override {
    // TODO:
    //    IMPLEMENT THIS!
  }

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

  // quantization
  template <typename K>
  DenseLayer<K> quantize(K quantization_bias, K quantization_scale) const {
    DenseLayer<K> new_layer(this->in_size_, out_size_);
    new_layer.quantization_bias = quantization_bias;
    new_layer.quantization_scale = quantization_scale;
    auto w = ::quantize<K>(weights_, quantization_bias, quantization_scale);
    new_layer.load_weights(w);
    auto b = ::quantize<K>(bias_, quantization_bias, quantization_scale);
    new_layer.load_bias(b);
    return new_layer;
  }

protected:
  uint32_t out_size_;
  Tensor<T> weights_;
  Tensor<T> bias_;
};

template <typename T> class ActivationLayer : public Layer<T> {
public:
  inline explicit ActivationLayer(const TensorSize &size)
      : Layer<T>(size, size) {}

  inline void forward(const Tensor<T> &in) {
    for (auto y = 0; y < this->out_.size.y; y++) {
      for (auto x = 0; x < this->out_.size.x; x++) {
        for (auto c = 0; c < this->out_.size.c; c++) {
          for (auto k = 0; k < this->out_.size.k; k++) {
            this->out_(y, x, c, k) = activate_function(in(y, x, c, k));
          }
        }
      }
    }
  }

  inline void forward_simd(const Tensor<T> &in) override {
    forward(in);
  }

  inline void forward_quantized(const Tensor<T> &in) override {
    for (auto y = 0; y < this->out_.size.y; y++) {
      for (auto x = 0; x < this->out_.size.x; x++) {
        for (auto c = 0; c < this->out_.size.c; c++) {
          for (auto k = 0; k < this->out_.size.k; k++) {
            this->out_(y, x, c, k) =
                activate_function_quantized(in(y, x, c, k));
          }
        }
      }
    }
  }

protected:
  inline virtual T activate_function(T value) { return value; }
  inline virtual T activate_function_quantized(T value) { return value; }
};

template <typename T> class ReLuActivationLayer : public ActivationLayer<T> {
public:
  inline explicit ReLuActivationLayer(const TensorSize &size)
      : ActivationLayer<T>(size) {}

  // quantization
  template <typename K>
  ReLuActivationLayer<K> quantize(K quantization_bias,
                                  K quantization_scale) const {
    ReLuActivationLayer<K> new_layer(this->in_size_);
    new_layer.quantization_bias = quantization_bias;
    new_layer.quantization_scale = quantization_scale;
    // noop
    return new_layer;
  }

protected:
  inline T activate_function(T value) override { return value > 0 ? value : 0; }

  inline T activate_function_quantized(T value) override {
    // TODO:
    //    IMPLEMENT THIS!
    return 0;
  }
};

template <typename T> class SigmoidActivationLayer : public ActivationLayer<T> {
public:
  inline explicit SigmoidActivationLayer(const TensorSize &size)
      : ActivationLayer<T>(size) {}

protected:
  inline T activate_function(T value) override { return 0; }
};

#endif // UDNN_LAYER_HH
