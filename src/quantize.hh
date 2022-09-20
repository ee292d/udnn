#ifndef INCLUDE_QUANTIZE_HH
#define INCLUDE_QUANTIZE_HH

#include "tensor.hh"

// to see more details about fixed point:
//    https://vanhunteradams.com/FixedPoint/FixedPoint.html

template <typename T> constexpr float fix2float(T value) {
  float scale = 0;
  if constexpr (std::is_same_v<T, int8_t>) {
    scale = 16.f;
  } else {
    scale = 256.f;
  }
  auto res = static_cast<float>(value) / scale;

  return res;
}

template <typename T> constexpr T float2fix(float value) {
  float scale = 0;
  if constexpr (std::is_same_v<T, int8_t>) {
    scale = 16.f;
  } else {
    scale = 256.f;
  }
  // clamping
  constexpr auto max = fix2float(std::numeric_limits<T>::max());
  constexpr auto min = fix2float(std::numeric_limits<T>::min());
  value = std::max(min, std::min(value, max));
  auto mult = value * scale;
  auto res = static_cast<T>(mult);

  return res;
}

template <typename T>
T quantize(float value, T quantize_bias, T quantize_scale) {
  // TODO:
  //  implement this
  return 0;
}

template <typename T, typename K>
Tensor<T> quantize(const Tensor<K> &tensor, T quantize_bias, T quantize_scale) {
  Tensor<T> res(tensor.size);
  for (auto k = 0; k < tensor.size.k; k++) {
    for (auto c = 0; c < tensor.size.c; c++) {
      for (auto y = 0; y < tensor.size.y; y++) {
        for (auto x = 0; x < tensor.size.x; x++) {
          auto v = quantize(tensor(y, x, c, k), quantize_bias, quantize_scale);
          res(y, x, c, k) = static_cast<K>(v);
        }
      }
    }
  }

  return res;
}

template <typename T>
float unquantize(T value, T quantize_bias, T quantize_scale) {
  // TODO:
  //   implement this
  return 0;
}

template <typename T>
Tensor<float> unquantize(const Tensor<T> &tensor, T quantize_bias,
                         T quantize_scale) {
  Tensor<float> res(tensor.size);
  for (auto k = 0; k < tensor.size.k; k++) {
    for (auto c = 0; c < tensor.size.c; c++) {
      for (auto y = 0; y < tensor.size.y; y++) {
        for (auto x = 0; x < tensor.size.x; x++) {
          auto v =
              unquantize(tensor(y, x, c, k), quantize_bias, quantize_scale);
          res(y, x, c, k) = v;
        }
      }
    }
  }

  return res;
}

template <typename T> T fix_mult(T a, T b) {
  if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
    return a * b;
  } else {
    return (a >> (sizeof(T) * 2)) * (b >> (sizeof(T) * 2));
  }
}

template <typename T>
T quantize_add(T a, T b, T quantize_bias, T quantize_scale) {
  // TODO:
  //  implement this
  return 0;
}

template <typename T>
T quantize_mult(T a, T b, T quantize_bias, T quantize_scale) {
  // TODO:
  //  implement this
  return 0;
}

template <typename T>
T quantize_add_opt(T a, T b, T quantize_bias, T quantize_scale,
                   T coefficient1) {
  // TODO:
  //  implement this
  //  add more precomputed coefficients as you see necessary
  return 0;
}

template <typename T>
T quantize_mult_opt(T a, T b, T quantize_bias, T quantize_scale, T coefficient1,
                    T coefficient2) {
  // TODO:
  //  implement this
  //  add more precomputed coefficients as you see necessary
  return 0;
}

#endif // INCLUDE_QUANTIZE_HH
