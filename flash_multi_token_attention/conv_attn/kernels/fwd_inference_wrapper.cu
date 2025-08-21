#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <tuple>

// Common utilities
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#include "fwd_inference.cu"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> flash_conv_attention_inference(
    torch::Tensor Q,     // [batch, heads, q_len, d_head] - query tokens (q_len <= 16)
    torch::Tensor K,     // [batch, heads, seq_len, d_head]
    torch::Tensor V,     // [batch, heads, seq_len, d_head]
    torch::Tensor conv_kernel,  // [heads, kx, ky]
    int k_splits,        // Number of splits for K dimension
    float scale,
    int pad_q,
    int pad_k,
    bool causal,
    int num_k_blocks
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
    const int q_seq_len = Q.size(2);  // Can be <= 16
    const int d_head = Q.size(3);
    const int seq_len_k = K.size(2);
    
    TORCH_CHECK(q_seq_len <= 16 && q_seq_len > 0, "Q sequence length must be between 1 and 16 for inference");
    
    // Pad Q to 16 tokens if needed
    torch::Tensor Q_padded;
    if (q_seq_len < 16) {
        // Create zero-padded tensor
        Q_padded = torch::zeros({batch_size, num_heads, 16, d_head}, 
                               torch::dtype(torch::kBFloat16).device(Q.device()));
        // Copy original Q to the beginning
        Q_padded.narrow(2, 0, q_seq_len).copy_(Q);
    } else {
        Q_padded = Q;
    }
    TORCH_CHECK(K.size(0) == batch_size, "K batch size must match Q");
    TORCH_CHECK(K.size(1) == num_heads, "K num_heads must match Q");
    TORCH_CHECK(K.size(3) == d_head, "K d_head must match Q");
    TORCH_CHECK(V.size(0) == batch_size, "V batch size must match Q");
    TORCH_CHECK(V.size(1) == num_heads, "V num_heads must match Q");
    TORCH_CHECK(V.size(2) == seq_len_k, "V seq_len must match K");
    TORCH_CHECK(V.size(3) == d_head, "V d_head must match Q");
    
    const int kx = conv_kernel.size(1);
    const int ky = conv_kernel.size(2);
    
    TORCH_CHECK(k_splits > 0, "k_splits must be positive");
    TORCH_CHECK(k_splits <= seq_len_k, "k_splits cannot be larger than seq_len_k");
    
    // Create output tensors with k_splits dimension
    auto O = torch::zeros({batch_size, num_heads, k_splits, d_head}, 
                         torch::dtype(torch::kFloat).device(Q.device()));
    auto M = torch::zeros({batch_size, num_heads, k_splits}, 
                         torch::dtype(torch::kFloat).device(Q.device()));
    auto L = torch::zeros({batch_size, num_heads, k_splits}, 
                         torch::dtype(torch::kFloat).device(Q.device()));
    
    #define LAUNCH_INFERENCE_KERNEL(KX, KY, HEAD_DIM) \
        launch_inference_kernel<KX, KY, HEAD_DIM>( \
            reinterpret_cast<const __nv_bfloat16*>(Q_padded.data_ptr()), \
            reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()), \
            reinterpret_cast<const __nv_bfloat16*>(V.data_ptr()), \
            reinterpret_cast<const __nv_bfloat16*>(conv_kernel.data_ptr()), \
            reinterpret_cast<float*>(O.data_ptr()), \
            reinterpret_cast<float*>(M.data_ptr()), \
            reinterpret_cast<float*>(L.data_ptr()), \
            batch_size, num_heads, seq_len_k, k_splits, scale, pad_q, pad_k, causal, num_k_blocks \
        )
    
    if (kx == 3 && ky == 3) {
        if (d_head == 32) LAUNCH_INFERENCE_KERNEL(3, 3, 32);
        else if (d_head == 64) LAUNCH_INFERENCE_KERNEL(3, 3, 64);
        else if (d_head == 128) LAUNCH_INFERENCE_KERNEL(3, 3, 128);
        else TORCH_CHECK(false, "Unsupported head dimension: " + std::to_string(d_head));
    } else if (kx == 5 && ky == 5) {
        if (d_head == 32) LAUNCH_INFERENCE_KERNEL(5, 5, 32);
        else if (d_head == 64) LAUNCH_INFERENCE_KERNEL(5, 5, 64);
        else if (d_head == 128) LAUNCH_INFERENCE_KERNEL(5, 5, 128);
        else TORCH_CHECK(false, "Unsupported head dimension: " + std::to_string(d_head));
    } else if (kx == 6 && ky == 6) {
        if (d_head == 32) LAUNCH_INFERENCE_KERNEL(6, 6, 32);
        else if (d_head == 64) LAUNCH_INFERENCE_KERNEL(6, 6, 64);
        else if (d_head == 128) LAUNCH_INFERENCE_KERNEL(6, 6, 128);
        else TORCH_CHECK(false, "Unsupported head dimension: " + std::to_string(d_head));
    } else if (kx == 7 && ky == 7) {
        if (d_head == 32) LAUNCH_INFERENCE_KERNEL(7, 7, 32);
        else if (d_head == 64) LAUNCH_INFERENCE_KERNEL(7, 7, 64);
        else if (d_head == 128) LAUNCH_INFERENCE_KERNEL(7, 7, 128);
        else TORCH_CHECK(false, "Unsupported head dimension: " + std::to_string(d_head));
    } else if (kx == 11 && ky == 11) {
        if (d_head == 32) LAUNCH_INFERENCE_KERNEL(11, 11, 32);
        else if (d_head == 64) LAUNCH_INFERENCE_KERNEL(11, 11, 64);
        else if (d_head == 128) LAUNCH_INFERENCE_KERNEL(11, 11, 128);
        else TORCH_CHECK(false, "Unsupported head dimension: " + std::to_string(d_head));
    } else {
        TORCH_CHECK(false, "Unsupported kernel size: " + std::to_string(kx) + "x" + std::to_string(ky));
    }
    
    #undef LAUNCH_INFERENCE_KERNEL
    
    return std::make_tuple(O, M, L);
}