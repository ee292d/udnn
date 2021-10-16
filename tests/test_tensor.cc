#include "../extern/googletest/googletest/include/gtest/gtest.h"
#include "../src/tensor.hh"
#include <random>

#ifdef _WIN32
#include <fileapi.h>
constexpr char path_seq = '\\';
#else
constexpr char path_seq = '/';
#endif

std::string get_temp_dir() {
#ifdef _WIN32
  wchar_t buffer[512];
  auto rc = GetTempPathW(511, buffer);
  if (!rc || rc > 511) {
    return "";
  }
  return std::string(buffer);
#else
  return "./";
#endif
}

std::string get_temp_file(const std::string &filename) {
  return get_temp_dir() + path_seq + filename;
}


TEST(tensor, get_set) {
  Tensor<int8_t> t(2, 2, 1);
  t(0, 1, 0) = 1;
  EXPECT_EQ(t(0, 1, 0), 1);
  t(1, 1, 0) = 2;
  EXPECT_EQ(t(1, 1, 0), 2);
  EXPECT_THROW(t(0, 0, 1), std::range_error);
}

TEST(tensor, io) {
  Tensor<float> t(2, 3, 1);
  std::random_device r;
  // engine
  std::mt19937 e(r());
  std::uniform_real_distribution<> dist(-10, 10);
  for (uint32_t y = 0; y < 2; y++) {
    for (uint32_t x = 0; x < 3; x++) {
      t(y, x, 0) = dist(e);
    }
  }
  // dump to a file
  std::string filename = get_temp_file("tensor.dat");
  t.dump(filename);

  auto t_in = Tensor<float>::load(filename);
  EXPECT_EQ(t_in.size, t.size);
  EXPECT_EQ(t_in.stride(), t.stride());
}
