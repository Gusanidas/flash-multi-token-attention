#include <torch/extension.h>

// Forward declarations of wrapper functions
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> flash_conv_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor conv_kernel,
    float scale,
    int pad_q,
    int pad_k,
    bool causal
);

std::tuple<torch::Tensor, torch::Tensor> flash_conv_attention_dq_backward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor dO,
    torch::Tensor D,
    torch::Tensor L,
    torch::Tensor conv_kernel,
    float scale,
    int pad_q,
    int pad_k,
    bool causal
);

std::tuple<torch::Tensor, torch::Tensor> flash_conv_attention_dkdv_backward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor dO,
    torch::Tensor D,
    torch::Tensor L,
    torch::Tensor conv_kernel,
    float scale,
    int pad_q,
    int pad_k,
    bool causal
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Flash Convolution Attention CUDA extensions";
    
    m.def("forward", &flash_conv_attention_forward, "Flash Convolution Attention Forward",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("conv_kernel"),
          py::arg("scale"), py::arg("pad_q"), py::arg("pad_k"), py::arg("causal"));
    
    m.def("dq_backward", &flash_conv_attention_dq_backward, "Flash Convolution Attention dQ Backward",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("dO"), py::arg("D"), py::arg("L"),
          py::arg("conv_kernel"), py::arg("scale"), py::arg("pad_q"), py::arg("pad_k"), py::arg("causal"));
    
    m.def("dkdv_backward", &flash_conv_attention_dkdv_backward, "Flash Convolution Attention dK/dV Backward",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("dO"), py::arg("D"), py::arg("L"),
          py::arg("conv_kernel"), py::arg("scale"), py::arg("pad_q"), py::arg("pad_k"), py::arg("causal"));
}