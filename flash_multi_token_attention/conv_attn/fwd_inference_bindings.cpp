#include <torch/extension.h>

// Forward declaration of wrapper function
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> flash_conv_attention_inference(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor conv_kernel,
    int k_splits,
    float scale,
    int pad_q,
    int pad_k,
    bool causal,
    int num_k_blocks
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Flash Convolution Attention Inference CUDA extensions";
    
    m.def("forward_inference", &flash_conv_attention_inference, "Flash Convolution Attention Forward Inference",
          py::arg("Q"), py::arg("K"), py::arg("V"), py::arg("conv_kernel"),
          py::arg("k_splits"), py::arg("scale"), py::arg("pad_q"), py::arg("pad_k"), py::arg("causal"),
          py::arg("num_k_blocks"));
}