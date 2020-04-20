#include "../extern/googletest/googletest/include/gtest/gtest.h"
#include "../src/tensor.hh"

TEST(simd, transform) {
  constexpr int size = 8;
  auto array0 = Tensor<int8_t>::vector_type(size);
  for (int8_t i = 0; i < size; i++)
    array0[i] = i;
  auto array1 = Tensor<int8_t>::vector_type(size);
  for (int8_t i = 0; i < size; i++)
    array1[i] = -i;
  // the output has to have enough space, otherwise there will be segfault
  auto result = Tensor<int8_t>::vector_type(size);
  xsimd::transform(array0.begin(), array0.end(), array1.begin(), result.begin(),
                   [](auto const &a, auto const &b) { return a * b; });
  for (int8_t i = 0; i < size; i++) {
    EXPECT_EQ(result[i], -i * i);
  }
}

TEST(simd, add) {
  constexpr int array_size = 100;
  auto a = Tensor<double>::vector_type(array_size);
  auto b = Tensor<double>::vector_type(array_size);
  auto res = Tensor<double>::vector_type(array_size);

  // fill in numbers
  for (auto i = 0; i < array_size; i++) {
    a[i] = static_cast<double>(i);
    b[i] = static_cast<double>(i);
  }

  using b_type = xsimd::simd_type<double>;
  auto inc = Tensor<double>::simd_size();
  auto size = a.size();
  // size for which the vectorization is possible
  auto vec_size = size - size % inc;
  for (std::size_t i = 0; i < vec_size; i += inc) {
    b_type a_vec = xsimd::load_aligned(&a[i]);
    b_type b_vec = xsimd::load_aligned(&b[i]);
    b_type r_vec = a_vec + b_vec;
    xsimd::store_aligned(&res[i], r_vec);
  }
  // Remaining part that cannot be vectorize
  for (auto i = vec_size; i < size; ++i) {
    res[i] = (a[i] + b[i]) / 2;
  }

  for (auto i = 0; i < array_size; i++) {
    EXPECT_EQ(res[i], i * 2.0);
  }
}

TEST(simd, reduce) {
  constexpr int size = 101;
  auto array = Tensor<int16_t>::vector_type(size);
  for (int16_t i = 0; i < size; i++)
    array[i] = i;
  auto result = xsimd::reduce(array.begin(), array.end(), 0);
  EXPECT_EQ(result, 5050);
}

TEST(simd, max) {
  constexpr int size = 101;
  auto array = Tensor<int16_t>::vector_type(size);
  for (int16_t i = 0; i < size; i++)
    array[i] = i;
  auto result = xsimd::reduce(
      array.begin(), array.end(), array[0],
      [](const auto &a, const auto &b) { return xsimd::max(a, b); });
  EXPECT_EQ(result, size - 1);
}