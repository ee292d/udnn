#include "layer.hh"
#include <chrono>
#include <iostream>

// example benchmark code. feel free to modify to suit your need
auto constexpr num_run = 10;
auto constexpr num_filters = 16;

template <typename T>
void init_random_tensor(Tensor<T> &tensor, T min = 0, T max = 42) {
  auto const &size = tensor.size;
  for (auto x = 0u; x < size.x; x++) {
    for (auto y = 0u; y < size.y; y++) {
      for (auto k = 0u; k < size.k; k++) {
        for (auto c = 0; c < size.c; c++) {
          tensor(y, x, c, k) = 1;
        }
      }
    }
  }
}

template <typename T>
void init_random_weight(Layer<T> &layer, T min = 0, T max = 42) {
  auto const *weight = layer.get_weights();
  Tensor<T> t(weight->size);
  init_random_tensor(t, min, max);
  layer.load_weights(t);
}

template <typename T>
void init_random_bias(Layer<T> &layer, T min = 0, T max = 42) {
  auto const *bias = layer.get_bias();
  Tensor<T> t(bias->size);
  init_random_tensor(t, min, max);
  layer.load_bias(t);
}

int main(int argc, char *argv[]) {
  Tensor<int8_t> input(128, 128, 3);
  auto layer = Conv2DLayer<int8_t>(input.size, 3, num_filters);
  init_random_tensor(input);
  init_random_weight(layer);
  init_random_bias(layer);

  // start the clock
  auto start = std::chrono::system_clock::now();

  for (int i = 0u; i < num_run; i++) {
    layer.forward(input);
  }

  auto end = std::chrono::system_clock::now();

  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count();

  // print out the result
  std::cout << ms << std::endl;

  return EXIT_SUCCESS;
}