#include "model.hh"


void Model::add_layer(const std::string &name, LayerBase *layer) {
  // the name has to be unique
  if (names_.find(name) != names_.end())
    throw std::invalid_argument("A layer with name " + name +
                                " already exists");
  layer->name = name;
  if (!layers_.empty()) {
    // connect them together
    auto pre = layers_.back();
    pre->connect(layer);
  }
  layers_.emplace_back(layer);
}

void Model::predict(const TensorBase *tensor) {
  if (layers_.empty())
    return;
  auto layer = layers_.front();
  do {
    // we don't do any copy here
    layer->forward(tensor->ptr());
    tensor = layer->out_base();
    layer = layer->next();
  } while (layer != nullptr);
}

const TensorBase * Model::out() const {
  if (!layers_.empty())
    return layers_.back()->out_base();
  else
    return nullptr;
}

DType Model::out_type() const {
  // default float
  if (layers_.empty()) return DType::Float;
  return layers_.back()->out_type();
}