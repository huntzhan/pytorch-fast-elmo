#include "extension/elmo_character_encoder.h"
#include <stdexcept>

namespace cnt {

HighwayImpl::HighwayImpl(
    int64_t input_dim,
    int64_t num_layers,
    TorchActivationType activation)
    :
    input_dim_(input_dim),
    activation_(activation) {
  // Build layers.
  for (int64_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
    // Ouptut: concat(H, T).
    auto layer = torch::nn::Linear(input_dim_, input_dim_ * 2);
    // Initially biased towards carry behavior.
    layer->bias
        .detach()
        .narrow(0, input_dim_, input_dim_)
        .fill_(-1);
    // Note: Libtorch 1.0 doesn't support module list,
    //       so we need to register all layers explictly.
    register_module(
        "layers_" + std::to_string(layer_idx),
        layer);
    layers_.push_back(layer);
  }
}

torch::Tensor HighwayImpl::forward(torch::Tensor inputs) {
  auto cur_inputs = inputs;

  for (auto layer : layers_) {
    auto proj_inputs = layer(cur_inputs);

    auto transform = proj_inputs.narrow(-1, 0, input_dim_);
    auto transform_gate = proj_inputs.narrow(-1, input_dim_, input_dim_);

    transform = activation_(transform);
    transform_gate = torch::sigmoid(transform_gate);
    cur_inputs = transform_gate * transform + (1 - transform_gate) * cur_inputs;
  }
  return cur_inputs;
}

ElmoCharacterEncoderImpl::ElmoCharacterEncoderImpl(
    int64_t char_embedding_cnt,
    int64_t char_embedding_dim,
    ElmoCharacterEncoderFiltersType filters,
    std::string activation,
    int64_t num_highway_layers,
    int64_t output_dim) {
  // Build char embedding.
  char_embedding_ = torch::nn::Embedding(
      // Add offset 1 for padding.
      char_embedding_cnt + 1,
      char_embedding_dim);
  register_module("char_embedding", char_embedding_);

  // Build CNN.
  int64_t total_out_channels = 0;
  for (int64_t conv_idx = 0;
      conv_idx < static_cast<int64_t>(filters.size());
      ++conv_idx) {
    // Config.
    auto kernel_size = std::get<0>(filters[conv_idx]);
    auto out_channels = std::get<1>(filters[conv_idx]);

    total_out_channels += out_channels;

    auto conv_options =
        torch::nn::Conv1dOptions(
            char_embedding_dim,
            out_channels,
            kernel_size)
        // Explicitly set bias.
        .with_bias(true);

    // Build.
    auto conv = torch::nn::Conv1d(conv_options);
    register_module(
        "char_conv_" + std::to_string(conv_idx),
        conv);
    convolutions_.push_back(conv);
  }

  // Bind CNN activation.
  if (activation == "tanh") {
    activation_ = &torch::tanh;
  } else if (activation == "relu") {
    activation_ = &torch::relu;
  } else {
    throw std::invalid_argument("Invalid activation.");
  }

  // Build highway layers.
  highway_ = Highway(
      total_out_channels,
      num_highway_layers,
      // hardcoded as bilm-tf.
      &torch::relu);
  register_module("highway", highway_);

  // Build projection.
  output_proj_ = torch::nn::Linear(total_out_channels, output_dim);
  register_module("output_proj", output_proj_);
}

torch::Tensor ElmoCharacterEncoderImpl::forward(torch::Tensor inputs) {
  // Of shape `(*, char_embedding_dim, max_chars_per_token)`.
  auto char_embds = char_embedding_(inputs).transpose(1, 2);

  // Apply CNN.
  std::vector<torch::Tensor> conv_outputs(convolutions_.size());
  for (int64_t conv_idx = 0;
      conv_idx < static_cast<int64_t>(convolutions_.size());
      ++conv_idx) {
    // `(*, C_out, L_out)`
    auto convolved = convolutions_[conv_idx](char_embds);
    // `(*, C_out)`
    convolved = std::get<0>(torch::max(convolved, -1));  // NOLINT
    convolved = activation_(convolved);

    conv_outputs[conv_idx] = convolved;
  }
  // `(*, total_out_channels)`
  auto char_repr = torch::cat(conv_outputs, -1);

  // Apply highway.
  // `(*, total_out_channels)`
  char_repr = highway_(char_repr);

  // Apply output projection.
  // `(*, output_dim)`
  char_repr = output_proj_(char_repr);

  return char_repr;
}

}  // namespace cnt
