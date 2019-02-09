#include "extension/scalar_mix.h"
#include <stdexcept>

namespace cnt {

ScalarMixImpl::ScalarMixImpl(
    int64_t mixture_size,
    bool do_layer_norm,
    std::vector<double> initial_scalar_parameters,
    bool trainable)
    :
    mixture_size_(mixture_size),
    do_layer_norm_(do_layer_norm) {
  if (initial_scalar_parameters.empty()) {
    // Initialize with 1/n.
    initial_scalar_parameters.insert(
        initial_scalar_parameters.end(),
        mixture_size,
        1.0 / static_cast<double>(mixture_size));
  } else if (
      static_cast<int64_t>(initial_scalar_parameters.size()) != \
      mixture_size) {
    throw std::invalid_argument(
        "initial_scalar_parameters & mixture_size not match.");
  }

  // Build scalar_parameters & gamma.
  // scalar_parameters.
  for (int64_t idx = 0;
      idx < static_cast<int64_t>(initial_scalar_parameters.size());
      ++idx) {
    auto scalar = torch::zeros({1}, torch::dtype(torch::kFloat32));
    scalar.detach().fill_(initial_scalar_parameters[idx]);

    register_parameter("scalar_" + std::to_string(idx), scalar, trainable);
    scalar_parameters_.push_back(scalar);
  }
  // gamma.
  gamma_ = torch::ones({1}, torch::dtype(torch::kFloat32));
  register_parameter("gamma", gamma_, trainable);
}

// Inputs: tensor, broadcast_mask, num_elements_not_masked
//   - **tensor** of shape `(*, features)`:
//     where * means any number of additional dimensions
//   - **broadcast_mask** of shape `(*, 1)`:
//     where * means the dimensions of **tensor**.
//   - **num_elements_not_masked** of shape `(1,)`:
//     the number of valid elements
//
// Outputs: output
//   - **output** of shape `(*, features)`:
//     normalized **tensor**.
//
// Note: Masked elements in output tensor won't be zeros.
inline torch::Tensor apply_layer_norm(
    torch::Tensor tensor,
    torch::Tensor broadcast_mask,
    torch::Tensor num_elements_not_masked) {
  auto tensor_masked = tensor * broadcast_mask;
  auto mean = torch::sum(tensor_masked) / num_elements_not_masked;
  auto variance =
      torch::sum(
          torch::pow(
              (tensor_masked - mean) * broadcast_mask,
              2)) / num_elements_not_masked;
  return (tensor - mean) / torch::sqrt(variance + 1E-12);
}

// We assume 1. the shapes of `tensors` are identical.
//           2. the shape of `mask`, if `mask` is provided,
//              should match the prefix of the shape of `tensors`.
torch::Tensor ScalarMixImpl::forward(
    const std::vector<torch::Tensor> &tensors,
    torch::Tensor mask) {
  // Check the length of `tensors`.
  if (static_cast<int64_t>(tensors.size()) != mixture_size_) {
    throw std::invalid_argument(
        "tensors & mixture_size not match.");
  }
  // Check the mask.
  if (do_layer_norm_ && !mask.defined()) {
    if (tensors[0].dim() == 2) {
      // To handle the packed sequences.
      mask = torch::ones({tensors[0].size(0)});
    } else {
      throw std::invalid_argument(
          "do_layer_norm but mask is not defined.");
    }
  }

  auto normed_weights = torch::split(
      torch::softmax(
          torch::cat(scalar_parameters_),
          0),
      1);

  torch::Tensor broadcast_mask;
  torch::Tensor num_elements_not_masked;
  if (do_layer_norm_) {
    auto mask_float = mask.to(torch::kFloat32);
    broadcast_mask = mask_float.unsqueeze(-1);
    auto input_dim = tensors[0].size(-1);
    num_elements_not_masked = torch::sum(mask_float) * input_dim;
  }

  torch::Tensor total;
  for (int64_t idx = 0; idx < mixture_size_; ++idx) {
    auto tensor = tensors[idx];
    if (do_layer_norm_) {
      tensor = apply_layer_norm(
          tensor,
          broadcast_mask,
          num_elements_not_masked);
    }
    auto weighted_tensor = normed_weights[idx] * tensor;
    if (idx == 0) {
      total = weighted_tensor;
    } else {
      total += weighted_tensor;
    }
  }
  return gamma_ * total;
}

}  // namespace cnt
