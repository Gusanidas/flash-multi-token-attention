#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <tuple>

// Common utilities (inline instead of using common.h)
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Include the kernel and launcher implementation
#include "fwd.cu"

// PyTorch wrapper function
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> flash_conv_attention_forward(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor conv_kernel,
    float scale,
    int pad_q,
    int pad_k,
    bool causal
) {
    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);
    CHECK_INPUT(conv_kernel);
    
    TORCH_CHECK(Q.dtype() == torch::kBFloat16, "Q must be bfloat16");
    TORCH_CHECK(K.dtype() == torch::kBFloat16, "K must be bfloat16");
    TORCH_CHECK(V.dtype() == torch::kBFloat16, "V must be bfloat16");
    TORCH_CHECK(conv_kernel.dtype() == torch::kBFloat16, "conv_kernel must be bfloat16");
    
    const int batch_size = Q.size(0);
    const int num_heads = Q.size(1);
    const int seq_len = Q.size(2);
    const int d_head = Q.size(3);
    
    const int kx = conv_kernel.size(1);
    const int ky = conv_kernel.size(2);
    
    // Create output tensors
    auto O = torch::zeros_like(Q).to(torch::kFloat);
    auto M = torch::zeros({batch_size, num_heads, seq_len}, torch::dtype(torch::kFloat).device(Q.device()));
    auto L = torch::zeros({batch_size, num_heads, seq_len}, torch::dtype(torch::kFloat).device(Q.device()));
    
    // Macro-based kernel launch for different configurations
    #define LAUNCH_KERNEL(KX, KY, HEAD_DIM) \
        launch_forward_kernel<KX, KY, HEAD_DIM>( \
            reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()), \
            reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()), \
            reinterpret_cast<const __nv_bfloat16*>(V.data_ptr()), \
            reinterpret_cast<const __nv_bfloat16*>(conv_kernel.data_ptr()), \
            reinterpret_cast<float*>(O.data_ptr()), \
            reinterpret_cast<float*>(M.data_ptr()), \
            reinterpret_cast<float*>(L.data_ptr()), \
            batch_size, \
            num_heads, \
            seq_len, \
            scale, \
            pad_q, \
            pad_k, \
            causal \
        )
    
    // Dispatch based on kernel size and head dimension
    if (kx == 3 && ky == 3) {
        if (d_head == 32) LAUNCH_KERNEL(3, 3, 32);
        else if (d_head == 64) LAUNCH_KERNEL(3, 3, 64);
        else if (d_head == 128) LAUNCH_KERNEL(3, 3, 128);
        else TORCH_CHECK(false, "Unsupported head dimension: " + std::to_string(d_head));
    } else if (kx == 5 && ky == 5) {
        if (d_head == 32) LAUNCH_KERNEL(5, 5, 32);
        else if (d_head == 64) LAUNCH_KERNEL(5, 5, 64);
        else if (d_head == 128) LAUNCH_KERNEL(5, 5, 128);
        else TORCH_CHECK(false, "Unsupported head dimension: " + std::to_string(d_head));
    } else if (kx == 6 && ky == 6) {
        if (d_head == 32) LAUNCH_KERNEL(6, 6, 32);
        else if (d_head == 64) LAUNCH_KERNEL(6, 6, 64);
        else if (d_head == 128) LAUNCH_KERNEL(6, 6, 128);
        else TORCH_CHECK(false, "Unsupported head dimension: " + std::to_string(d_head));
    } else if (kx == 7 && ky == 7) {
        if (d_head == 32) LAUNCH_KERNEL(7, 7, 32);
        else if (d_head == 64) LAUNCH_KERNEL(7, 7, 64);
        else if (d_head == 128) LAUNCH_KERNEL(7, 7, 128);
        else TORCH_CHECK(false, "Unsupported head dimension: " + std::to_string(d_head));
    } else if (kx == 11 && ky == 11) {
        if (d_head == 32) LAUNCH_KERNEL(11, 11, 32);
        else if (d_head == 64) LAUNCH_KERNEL(11, 11, 64);
        else if (d_head == 128) LAUNCH_KERNEL(11, 11, 128);
        else TORCH_CHECK(false, "Unsupported head dimension: " + std::to_string(d_head));
    } else {
        TORCH_CHECK(false, "Unsupported kernel size: " + std::to_string(kx) + "x" + std::to_string(ky));
    }
    
    #undef LAUNCH_KERNEL
    
    return std::make_tuple(O, M, L);
}