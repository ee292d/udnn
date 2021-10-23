#ifndef UDNN_TENSOR_HH
#define UDNN_TENSOR_HH

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <xsimd/xsimd.hpp>
#include <random>

struct TensorSize {
public:
  uint32_t y, x, c;
  uint32_t k = 1;

  [[nodiscard]] inline std::string str() const {
    return "(" + std::to_string(x) + ", " + std::to_string(y) + ", " +
           std::to_string(c) + ", " + std::to_string(k) + ")";
  }

  inline bool operator==(const TensorSize &size) const {
    return x == size.x && y == size.y && c == size.c && k == size.k;
  }

  inline bool operator!=(const TensorSize &size) const {
    return !(*this == size);
  }

  inline static TensorSize default_stride(const TensorSize &size) {
    return {size.x * size.c * size.k, size.c * size.k, size.k, 1};
  }
};

class TensorBase {
public:
  TensorSize size;
  // whether this tensor object owned the data
  bool owned = true;

  [[nodiscard]] virtual void *ptr() const = 0;

  [[nodiscard]] virtual size_t element_size() const = 0;
};

template <typename T> class Tensor : public TensorBase {
public:
  using vector_type =
      std::vector<T, xsimd::aligned_allocator<T, XSIMD_DEFAULT_ALIGNMENT>>;

  inline T *data() const {
    return owned ? const_cast<T *>(owned_data_.data()) : data_;
  }
  [[nodiscard]] inline void *ptr() const override { return reinterpret_cast<void *>(data()); }

  inline Tensor(void *data, const TensorSize &size, const TensorSize &stride,
                bool copy = false) {
    this->size = size;
    owned = copy;
    auto total_size = size.x * size.y * size.c * size.k;
    T *p = static_cast<T *>(data);
    if (copy) {
      this->stride_ = TensorSize::default_stride(size);
      owned_data_.resize(total_size);
      this->copy(size, stride, p);
      this->data_ = owned_data_.data();
    } else {
      // beware of the stride difference!
      this->data_ = p;
      stride_ = stride;
    }
  }

  inline Tensor(): stride_({0, 0, 0, 0}) {
    this->size = {0, 0, 0, 0};
  }

  inline Tensor(uint32_t y, uint32_t x, uint32_t c, uint32_t k = 1) {
    owned_data_.resize(x * y * c * k);
    size = {y, x, c, k};
    stride_ = TensorSize::default_stride(size);
    data_ = owned_data_.data();
  }

  inline explicit Tensor(const TensorSize &size)
      : Tensor(size.y, size.x, size.c, size.k) {
    stride_ = TensorSize::default_stride(size);
  }

  inline T &operator()(uint32_t y, uint32_t x, uint32_t c, uint32_t k = 0) {
    return get(y, x, c, k);
  }

  inline T operator()(uint32_t y, uint32_t x, uint32_t c,
                      uint32_t k = 0) const {
    return get(y, x, c, k);
  }

  inline T &get(uint32_t y, uint32_t x, uint32_t c, uint32_t k = 0) {
    if (x >= size.x || y >= size.y || c >= size.c || k >= size.k)
      throw std::range_error("Accessing tensor out of range");
    auto const st = stride();
    auto index = x * st.x + y * st.y + c * st.c + k * st.k;
    return owned ? owned_data_[index] : data_[index];
  }

  inline T get(uint32_t y, uint32_t x, uint32_t c, uint32_t k = 0) const {
    if (x >= size.x || y >= size.y || c >= size.c || k >= size.k)
      throw std::range_error("Accessing tensor out of range");
    auto const st = stride();
    auto index = x * st.x + y * st.y + c * st.c + k * st.k;
    return owned ? owned_data_[index] : data_[index];
  }

  inline auto begin() { return owned_data_.begin(); }
  inline auto end() { return owned_data_.end(); }

  inline Tensor(const Tensor<T> &other) : stride_(other.stride()) {
    size = other.size;
    // we will make an copy here
    auto total_size = size.x * size.y * size.c * size.k;
    owned_data_ = vector_type(other.data(), other.data() + total_size);
  }

  inline Tensor(const TensorSize &size, const TensorSize &stride)
      : stride_(stride) {
    this->size = size;
    auto total_size = size.x * size.y * size.c * size.k;
    owned_data_.resize(total_size);
  }

  inline void load(const Tensor<T> &other) {
    // total size has to match
    auto this_size = size.x * size.y * size.c * size.k;
    auto other_size = other.size.x * other.size.y * other.size.c * other.size.k;
    if (this_size != other_size) {
      throw std::invalid_argument("Tensor size does not match");
    }
    copy(other.size, other.stride(), other.data());
  }

  inline explicit Tensor(std::istream &in) { load(in); }

  inline void dump(std::ostream &out) {
    // dump the tensor
    out.write(reinterpret_cast<char *>(&size.y), sizeof(size.y));
    out.write(reinterpret_cast<char *>(&size.x), sizeof(size.x));
    out.write(reinterpret_cast<char *>(&size.c), sizeof(size.c));
    out.write(reinterpret_cast<char *>(&size.k), sizeof(size.k));
    out.write(reinterpret_cast<char *>(&stride_.y), sizeof(stride_.y));
    out.write(reinterpret_cast<char *>(&stride_.x), sizeof(stride_.x));
    out.write(reinterpret_cast<char *>(&stride_.c), sizeof(stride_.c));
    out.write(reinterpret_cast<char *>(&stride_.k), sizeof(stride_.k));
    auto total_size = size.y * size.x * size.c * size.k;
    for (uint32_t i = 0; i < total_size; i++)
      out.write(reinterpret_cast<char *>(&owned_data_[i]), sizeof(T));
  }

  /// dump the tensor to a file
  /// \param filename
  inline void dump(const std::string &filename) {
    std::ofstream out(filename, std::ios::binary);
    dump(out);
  }

  /// Load a tensor from a file
  /// \param filename
  /// \return
  inline static Tensor<T> load(const std::string &filename) {
    std::ifstream in(filename, std::ios::binary);
    return Tensor<T>(in);
  }

  /// Load a tensor from a file and copy its content to the caller
  /// \param filename
  inline void load_from_file(const std::string &filename) {
    std::ifstream in(filename, std::ios::binary);
    // the size before the load
    auto pre_size = size;
    load(in);
    if (pre_size != size)
      throw std::invalid_argument("Tensor size doesn't match");
  }

  inline void load(std::istream &in) {
    // load the tensor from stream
    // size
    // has to be binary mode. if it's string mode then this method won't work
    in.read(reinterpret_cast<char *>(&size.y), sizeof(size.y));
    in.read(reinterpret_cast<char *>(&size.x), sizeof(size.x));
    in.read(reinterpret_cast<char *>(&size.c), sizeof(size.c));
    in.read(reinterpret_cast<char *>(&size.k), sizeof(size.k));
    in.read(reinterpret_cast<char *>(&stride_.y), sizeof(stride_.y));
    in.read(reinterpret_cast<char *>(&stride_.x), sizeof(stride_.x));
    in.read(reinterpret_cast<char *>(&stride_.c), sizeof(stride_.c));
    in.read(reinterpret_cast<char *>(&stride_.k), sizeof(stride_.k));
    auto total_size = size.y * size.x * size.c * size.k;
    owned_data_.resize(total_size);
    for (uint32_t i = 0; i < total_size; i++)
      in.read(reinterpret_cast<char *>(&owned_data_[i]), sizeof(T));
  }

  // the shape is height, width, channel
  // we have channel last implementation, which is the default for tf
  [[nodiscard]] inline TensorSize stride() const { return stride_; }

  [[nodiscard]] inline size_t element_size() const override { return sizeof(T); }

  inline static std::size_t simd_size() { return xsimd::simd_type<T>::size; }

  // randomize the content
  void randomize(T min, T max) {
    std::random_device rd;
    std::mt19937 engine(rd());

    auto const total_size = size.x * size.y * size.c * size.k;
    if constexpr (std::is_same<T, float>::value || std::is_same<T, double>::value) {
      std::uniform_real_distribution<> d(min, max);
      for (auto i = 0; i < total_size; i++) {
        data_[i] = d(engine);
      }
    } else {
      std::uniform_int_distribution<> d(min, max);
      for (auto i = 0; i < total_size; i++) {
        data_[i] = d(engine);
      }
    }
  }

private:
  vector_type owned_data_;
  TensorSize stride_;

  T *data_ = nullptr;

  inline void copy(const TensorSize &other_size, const TensorSize &other_stride,
                   T *data) {
    auto this_stride = stride_;
    uint32_t yy = 0, xx = 0, cc = 0, kk = 0;
    for (uint32_t y = 0; y < size.y; y++) {
      for (uint32_t x = 0; x < size.x; x++) {
        for (uint32_t c = 0; c < size.c; c++) {
          for (uint32_t k = 0; k < size.k; k++) {

            if (kk >= other_size.k) {
              ++cc;
              kk = 0;
            }
            if (cc >= other_size.c) {
              ++xx;
              cc = 0;
            }
            if (xx >= other_size.x) {
              ++yy;
              xx = 0;
            }
            auto p_index = xx * other_stride.x + yy * other_stride.y +
                           cc * other_stride.c + kk * other_stride.k;
            kk++;
            auto this_index = x * this_stride.x + y * this_stride.y +
                              c * this_stride.c + k * this_stride.k;
            owned_data_[this_index] = data[p_index];
          }
        }
      }
    }
  }
};

#endif // UDNN_TENSOR_HH
