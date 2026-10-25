#include <torch/extension.h>

torch::Tensor naive_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v);
torch::Tensor fa1_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v);
torch::Tensor fa2_forward(torch::Tensor q, torch::Tensor k, torch::Tensor v);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("naive_forward", &naive_forward, "Naive attention: S and P round-trip through HBM");
  m.def("fa1_forward", &fa1_forward, "FlashAttention v1 forward (outer loop over K/V tiles)");
  m.def("fa2_forward", &fa2_forward, "FlashAttention v2 forward (outer loop over Q tiles)");
}
