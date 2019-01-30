#ifndef EXTENSION_SCALAR_MIX_H_
#define EXTENSION_SCALAR_MIX_H_

#include <torch/extension.h>
#include <vector>

namespace cnt {

struct ScalarMix : torch::nn::Module {
  ScalarMix(
      int64_t mixture_size,
      bool do_layer_norm,
      std::vector<double> initial_scalar_parameters,
      bool trainable);

  torch::Tensor forward(
      const std::vector<torch::Tensor> &tensors,
      torch::Tensor mask);

  torch::Tensor forward(
      const std::vector<torch::Tensor> &tensors);

  int64_t mixture_size_ = -1;
  bool do_layer_norm_ = false;

  std::vector<torch::Tensor> scalar_parameters_ = {};
  torch::Tensor gamma_;
};

}  // namespace cnt

#endif  // EXTENSION_SCALAR_MIX_H_
