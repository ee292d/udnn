#include "../extern/googletest/googletest/include/gtest/gtest.h"
#include "../src/udnn.hh"

TEST(quantize, mult) {
  auto a = float2fix<int16_t>(2.0);
  auto b = float2fix<int16_t>(2.0);
  auto x = quantize<int16_t>(4.0, a, b);
  auto y = quantize<int16_t>(6.0, a, b);
  auto res = quantize_mult<int16_t>(x, y, a, b);
  auto res_f = unquantize<int16_t>(res, a, b);
  EXPECT_EQ(res_f, 24.0);
}