#ifndef UDNN_MODEL_HH
#define UDNN_MODEL_HH

#include "layer.hh"
#include <unordered_set>

class Model {
public:
  void add_layer(const std::string &name, LayerBase *layer);

  void predict(const TensorBase *tensor);
  void predict_simd(const TensorBase *tensor);

  const TensorBase *out() const;

  DType out_type() const;

  bool quantized = false;

private:
  std::vector<LayerBase *> layers_;

  std::unordered_set<std::string> names_;

};

#endif // UDNN_MODEL_HH
