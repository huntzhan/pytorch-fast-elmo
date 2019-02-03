#include "extension/elmo_character_encoder.h"
#include "extension/scalar_mix.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  torch::python::bind_module<cnt::ElmoCharacterEncoder>(
      m, "ElmoCharacterEncoder")

      .def(
          py::init<
              int64_t, int64_t,
              cnt::ElmoCharacterEncoderFiltersType,
              std::string,
              int64_t, int64_t>(),
          // Required.
          py::arg("char_embedding_cnt"),
          py::arg("char_embedding_dim"),
          py::arg("filters"),
          py::arg("activation"),
          py::arg("num_highway_layers"),
          py::arg("output_dim"))

      .def(
          "cuda",
          [](cnt::ElmoCharacterEncoder& module, int64_t device) {
            module.to("cuda:" + std::to_string(device));
          })

      .def(
          "__call__",
          (
              torch::Tensor
              (cnt::ElmoCharacterEncoder::*)
              (torch::Tensor)
          )
              &cnt::ElmoCharacterEncoder::forward);

  torch::python::bind_module<cnt::ScalarMix>(
      m, "ScalarMix")

      .def(
          py::init<
              int64_t,
              bool,
              std::vector<double>,
              bool>(),
          // Required.
          py::arg("mixture_size"),
          // Optional.
          py::arg("do_layer_norm") = false,
          py::arg("initial_scalar_parameters") = std::vector<double>(),
          py::arg("trainable") = true)

      .def(
          "cuda",
          [](cnt::ScalarMix& module, int64_t device) {
            module.to("cuda:" + std::to_string(device));
          })

      .def(
          "__call__",
          (
              torch::Tensor
              (cnt::ScalarMix::*)
              (const std::vector<torch::Tensor> &, torch::Tensor)
          )
              &cnt::ScalarMix::forward)

      .def(
          "__call__",
          (
              torch::Tensor
              (cnt::ScalarMix::*)
              (const std::vector<torch::Tensor> &)
          )
              &cnt::ScalarMix::forward);
}
