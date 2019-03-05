#include "extension/elmo_character_encoder.h"
#include "extension/scalar_mix.h"

template <typename ModuleType, typename... Extra>
py::class_<ModuleType, Extra...> patch_methods(
    py::class_<ModuleType, Extra...> module) {
  module.attr("cuda") = nullptr;
  module.def(
      "cuda",
      [](ModuleType& module, torch::optional<int64_t> device) {
        if (device.has_value()) {
          module.to("cuda:" + std::to_string(device.value()));
        } else {
          module.to(at::kCUDA);
        }
        return module;
      });
  module.def(
      "cuda",
      [](ModuleType& module) {
        module.to(at::kCUDA);
        return module;
      });

  module.attr("cpu") = nullptr;
  module.def(
      "cpu",
      [](ModuleType& module) {
        module.to(at::kCPU);
        return module;
      });

  return module;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  patch_methods(
      torch::python::bind_module<cnt::ElmoCharacterEncoderImpl>(
          m, "ElmoCharacterEncoder"))

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
          "__call__",
          &cnt::ElmoCharacterEncoderImpl::forward);

  patch_methods(
      torch::python::bind_module<cnt::ScalarMixImpl>(
          m, "ScalarMix"))

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
          "__call__",
          &cnt::ScalarMixImpl::forward)

      .def(
          "__call__",
          [](cnt::ScalarMixImpl& module,
              const std::vector<torch::Tensor> &tensors) {
            return module.forward(tensors, torch::Tensor());
          });
}
