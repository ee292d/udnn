#include "../extern/googletest/googletest/include/gtest/gtest.h"
#include "../src/model.hh"

TEST(model, next) {
  Model m;

  DenseLayer<int8_t> fc_layer({1, 4, 1}, 2);
  ReLuActivationLayer<int8_t> relu({1, 2, 1});

  // add layers
  m.add_layer("fc", &fc_layer);
  EXPECT_NO_THROW(m.add_layer("relu", &relu));
}

TEST(model, predict) {
  Model m;
  DenseLayer<int8_t> fc_layer({1, 4, 1}, 2);
  ReLuActivationLayer<int8_t> relu({1, 2, 1});

  // add layers
  m.add_layer("fc", &fc_layer);
  m.add_layer("relu", &relu);
  Tensor<int8_t> in(1, 4, 1);
  EXPECT_NO_THROW(m.predict(&in));
  auto out = m.out();
  EXPECT_EQ(out->size, (TensorSize{1, 2, 1}));
}