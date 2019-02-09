#ifndef EXTENSION_ELMO_CHARACTER_ENCODER_H_
#define EXTENSION_ELMO_CHARACTER_ENCODER_H_

#include <torch/extension.h>
#include <vector>
#include <tuple>
#include <string>

namespace cnt {

using TorchActivationType =
    torch::Tensor (*)(const torch::Tensor &);
using ElmoCharacterEncoderFiltersType =
    std::vector<std::tuple<int64_t, int64_t>>;

struct HighwayImpl : torch::nn::Module {
  // https://arxiv.org/abs/1505.00387
  HighwayImpl(
      int64_t input_dim,
      int64_t num_layers,
      TorchActivationType activation);

  torch::Tensor forward(torch::Tensor inputs);

  int64_t input_dim_ = -1;
  std::vector<torch::nn::Linear> layers_ = {};
  TorchActivationType activation_ = nullptr;
};

TORCH_MODULE(Highway);

struct ElmoCharacterEncoderImpl : torch::nn::Module {
  ElmoCharacterEncoderImpl(
      // Char embedding.
      int64_t char_embedding_cnt,
      int64_t char_embedding_dim,

      // CNN filters: [[<kernel_size>, <out_channels>], ...]
      // Example: [[1, 4], [2, 8], [3, 16], [4, 32], [5, 64]]
      ElmoCharacterEncoderFiltersType filters,
      // CNN activation (supports "relu", "tanh").
      std::string activation,

      // The number of highways.
      int64_t num_highway_layers,
      // The final projection size.
      int64_t output_dim);

  // Inputs: inputs
  //   - **inputs** of shape `(*, max_characters_per_token)`:
  //     tensor of `PackedSequence.data`.
  //
  // Outputs: output
  //   - **output** of shape `(*, output_dim)`:
  //     tensor containing the representations of character.
  //
  // Note: Different to AllenNLP's implementation,
  //       BOS/EOS will not be injected here.
  torch::Tensor forward(torch::Tensor inputs);

  // Char embedding.
  torch::nn::Embedding char_embedding_ = nullptr;

  // CNN.
  std::vector<torch::nn::Conv1d> convolutions_ = {};
  TorchActivationType activation_ = nullptr;

  // Highway.
  Highway highway_ = nullptr;

  // Output projection.
  torch::nn::Linear output_proj_ = nullptr;
};

TORCH_MODULE(ElmoCharacterEncoder);

}  // namespace cnt

#endif  // EXTENSION_ELMO_CHARACTER_ENCODER_H_
