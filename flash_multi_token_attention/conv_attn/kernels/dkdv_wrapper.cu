#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

// Common utilities TODO: use common.h?
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#include "dkdv_kernel.cu"

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
) {
    CHECK_INPUT(Q);
    CHECK_INPUT(K);
    CHECK_INPUT(V);
    CHECK_INPUT(dO);
    CHECK_INPUT(D);
    CHECK_INPUT(L);
    CHECK_INPUT(conv_kernel);
    
    TORCH_CHECK(Q.dtype() == torch::kBFloat16, "Q must be bfloat16");
    TORCH_CHECK(K.dtype() == torch::kBFloat16, "K must be bfloat16");
    TORCH_CHECK(V.dtype() == torch::kBFloat16, "V must be bfloat16");
    TORCH_CHECK(dO.dtype() == torch::kBFloat16, "dO must be bfloat16");
    TORCH_CHECK(D.dtype() == torch::kBFloat16, "D must be bfloat16");
    TORCH_CHECK(L.dtype() == torch::kBFloat16, "L must be bfloat16");
    TORCH_CHECK(conv_kernel.dtype() == torch::kBFloat16, "conv_kernel must be bfloat16");
    
    const int batch_size = Q.size(0);
    const int num_heads = Q.size(1);
    const int seq_len = Q.size(2);
    const int d_head = Q.size(3);
    
    const int kx = conv_kernel.size(1);
    const int ky = conv_kernel.size(2);
    
    auto dK = torch::zeros({batch_size, num_heads, seq_len, d_head}, 
                          torch::dtype(torch::kFloat32).device(Q.device()));
    auto dV = torch::zeros({batch_size, num_heads, seq_len, d_head}, 
                          torch::dtype(torch::kFloat32).device(Q.device()));
    
    #define LAUNCH_DKDV_KERNEL(KX, KY, HEAD_DIM) \
        launch_dkdv_backward_kernel<KX, KY, HEAD_DIM>(Q, K, V, dO, D, L, conv_kernel, dK, dV, scale, pad_q, pad_k, causal)
    
    if (kx == 3 && ky == 3) {
        if (d_head == 32) LAUNCH_DKDV_KERNEL(3, 3, 32);
        else if (d_head == 64) LAUNCH_DKDV_KERNEL(3, 3, 64);
        else if (d_head == 128) LAUNCH_DKDV_KERNEL(3, 3, 128);
        else TORCH_CHECK(false, "Unsupported head dimension: " + std::to_string(d_head));
    } else if (kx == 5 && ky == 5) {
        if (d_head == 32) LAUNCH_DKDV_KERNEL(5, 5, 32);
        else if (d_head == 64) LAUNCH_DKDV_KERNEL(5, 5, 64);
        else if (d_head == 128) LAUNCH_DKDV_KERNEL(5, 5, 128);
        else TORCH_CHECK(false, "Unsupported head dimension: " + std::to_string(d_head));
    } else if (kx == 6 && ky == 6) {
        if (d_head == 32) LAUNCH_DKDV_KERNEL(6, 6, 32);
        else if (d_head == 64) LAUNCH_DKDV_KERNEL(6, 6, 64);
        else if (d_head == 128) LAUNCH_DKDV_KERNEL(6, 6, 128);
        else TORCH_CHECK(false, "Unsupported head dimension: " + std::to_string(d_head));
    } else if (kx == 7 && ky == 7) {
        if (d_head == 32) LAUNCH_DKDV_KERNEL(7, 7, 32);
        else if (d_head == 64) LAUNCH_DKDV_KERNEL(7, 7, 64);
        else if (d_head == 128) LAUNCH_DKDV_KERNEL(7, 7, 128);
        else TORCH_CHECK(false, "Unsupported head dimension: " + std::to_string(d_head));
    } else if (kx == 11 && ky == 11) {
        if (d_head == 32) LAUNCH_DKDV_KERNEL(11, 11, 32);
        else if (d_head == 64) LAUNCH_DKDV_KERNEL(11, 11, 64);
        else if (d_head == 128) LAUNCH_DKDV_KERNEL(11, 11, 128);
        else TORCH_CHECK(false, "Unsupported head dimension: " + std::to_string(d_head));
    } else {
        TORCH_CHECK(false, "Unsupported kernel size: " + std::to_string(kx) + "x" + std::to_string(ky));
    }
    
    #undef LAUNCH_DKDV_KERNEL
    
    return std::make_tuple(dK, dV);
}