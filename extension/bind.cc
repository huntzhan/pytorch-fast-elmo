#include "extension/elmo_character_encoder.h"
#include "extension/scalar_mix.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  torch::python::bind_module<cnt::ElmoCharacterEncoderImpl>(
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
          [](cnt::ElmoCharacterEncoderImpl& module, int64_t device) {
            module.to("cuda:" + std::to_string(device));
          })

      .def(
          "__call__",
          &cnt::ElmoCharacterEncoderImpl::forward);

  torch::python::bind_module<cnt::ScalarMixImpl>(
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
          [](cnt::ScalarMixImpl& module, int64_t device) {
            module.to("cuda:" + std::to_string(device));
          })

      .def(
          "__call__",
          &cnt::ScalarMixImpl::forward)

      .def(
          "__call__",
          [](cnt::ScalarMixImpl& module,
              const std::vector<torch::Tensor> &tensors) {
            return module.forward(tensors, torch::Tensor());
          });
}
