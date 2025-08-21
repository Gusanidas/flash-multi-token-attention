#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>
#include <cstdio>
#include <cuda_bf16.h>
#include <mma.h>
#include <cooperative_groups.h>
#include "../../common/wmma_utils.cuh"
#include "../../common/conv_utils.cuh"
#include "../../common/kernels/common.h"
#include "../../common/print_matrix.cuh"

using namespace nvcuda;
using namespace cooperative_groups;


// Backward kernel for computing dQ
// Uses similar structure to forward kernel but computes gradients
// D_HEAD, KX and KY can all vary as template parameters
template<int BLOCK_SIZE_X, int BLOCK_SIZE_Y, int KX, int KY, int D_HEAD>
__global__ void dq_kernel(
    const __nv_bfloat16* __restrict__ Q,          // [batch, heads, seq_len, d_head]
    const __nv_bfloat16* __restrict__ K,          // [batch, heads, seq_len, d_head]
    const __nv_bfloat16* __restrict__ V,          // [batch, heads, seq_len, d_head]
    const __nv_bfloat16* __restrict__ dO,         // [batch, heads, seq_len, d_head] - gradient of output
    const __nv_bfloat16* __restrict__ D,          // [batch, heads, seq_len] - D values from separate computation
    const __nv_bfloat16* __restrict__ L,          // [batch, heads, seq_len] - L = M + log(sum) from forward
    const __nv_bfloat16* __restrict__ conv_kernel, // [heads, kx, ky]
    float* __restrict__ dQ,                       // [batch, heads, seq_len, d_head] - output gradient
    float* __restrict__ dconv_kernel,             // [heads, kx, ky] - convolution kernel gradient
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const float scale,
    const int pad_q,
    const int pad_k,
    const bool causal
) {
    
    // 2D thread indices
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    const int tid_linear = tid_y * BLOCK_SIZE_X + tid_x;
    
    // Get batch and head indices from grid
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    
    // Each block handles one Q block
    const int q_block_idx = blockIdx.x;
    const int reduced_block_size = BLOCK_SIZE_X - 2*pad_q;
    const int q_start = q_block_idx * reduced_block_size-2*pad_q;
    const int q_idx = q_start + tid_x;
    
    // Shared memory allocations
    extern __shared__ float shared_mem[];
    
    float* shared_mem_float = shared_mem;
    
    float* s_QK = shared_mem_float;                                                      // [BLOCK_SIZE_X + 2, BLOCK_SIZE_X] - SEPARATE
    float* s_dQ_accum = s_QK;
    constexpr int s_dQ_accum_size = (D_HEAD + 4) * BLOCK_SIZE_X;
    constexpr int s_QK_size = (BLOCK_SIZE_X + 2) * BLOCK_SIZE_X;
    constexpr int dQ_accum_or_QK_size = s_dQ_accum_size>s_QK_size?s_dQ_accum_size:s_QK_size;
    float* s_D = s_QK + dQ_accum_or_QK_size;                             // [BLOCK_SIZE_X]
    float* s_row_sum = s_D + BLOCK_SIZE_X;                                              // [BLOCK_SIZE_X]
    float* s_dP = s_row_sum + BLOCK_SIZE_X;                                             // [BLOCK_SIZE_X, BLOCK_SIZE_X]
    
    // Now bfloat16 arrays 
    __nv_bfloat16* s_Kv = (__nv_bfloat16*)(s_dP + BLOCK_SIZE_X * BLOCK_SIZE_X); // [BLOCK_SIZE_X, D_HEAD + 4] - for K and V
    __nv_bfloat16* s_dO = s_Kv + BLOCK_SIZE_X * (D_HEAD + 4);                         // [BLOCK_SIZE_X, D_HEAD + 4] - separate space for dO
    __nv_bfloat16* s_Q = s_dO + BLOCK_SIZE_X * (D_HEAD + 4);                          // [BLOCK_SIZE_X, D_HEAD + 4] - shared with P
    __nv_bfloat16* s_P = s_Q;                                                          // [BLOCK_SIZE_X, BLOCK_SIZE_X] - shares with Q
    constexpr int s_P_size = BLOCK_SIZE_X * BLOCK_SIZE_X;
    constexpr int s_Q_size = BLOCK_SIZE_X * (D_HEAD + 4);
    constexpr int P_or_Q_size = s_Q_size>s_P_size?s_Q_size:s_P_size;
    __nv_bfloat16* s_conv_kernel = s_Q + P_or_Q_size;                  // [KX, KY]
    
    float* s_dconv_kernel_accum = (float*)(s_conv_kernel + 4*((KX*KY + 4)/4));
    
    // Additional shared memory aliases (declared here but used later)
    float* s_dS = s_dP;                                                                // [BLOCK_SIZE_X, BLOCK_SIZE_X] - reuses s_dP space
    __nv_bfloat16* s_dQK = s_P;                                                        // [BLOCK_SIZE_X, BLOCK_SIZE_X] - reuses s_P space
    
    // Calculate base offsets for this batch and head
    const int qkv_offset = (batch_idx * num_heads + head_idx) * seq_len * D_HEAD;
    const int conv_kernel_offset = head_idx * KX * KY;
    const int ml_offset = (batch_idx * num_heads + head_idx) * seq_len;
    
    // Warp ID for WMMA instructions
    const int warp_id = tid_linear / 32;
    
    // Calculate group size based on block size for optimal memory access
    constexpr int GROUP_SIZE = 8;  // TODO: this should be a template parameter
    constexpr int elements_per_thread = BLOCK_SIZE_X / GROUP_SIZE;  // = 4 for BLOCK_SIZE_X=32, = 8 for BLOCK_SIZE_X=64
    
    // Create cooperative groups for convolution
    thread_block block = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(block);
    thread_block_tile<GROUP_SIZE> row_group = tiled_partition<GROUP_SIZE>(warp);
    
    const int row_group_id = tid_linear / GROUP_SIZE;
    const int thread_in_group = tid_linear % GROUP_SIZE;
    
    
    const int kernel_size = KX * KY;
    for (int i = tid_linear; i < kernel_size; i += BLOCK_SIZE_X * BLOCK_SIZE_Y) {
        s_conv_kernel[i] = conv_kernel[conv_kernel_offset + i];
        s_dconv_kernel_accum[i] = 0.0f; 
    }
    
    initialize_shared_memory<BLOCK_SIZE_X, BLOCK_SIZE_Y, D_HEAD>(
        nullptr, nullptr, nullptr, s_dQ_accum, tid_x, tid_y, tid_linear, -2000.0f, 0.0f, 1.0f
    );
    
    load_matrix<BLOCK_SIZE_Y, D_HEAD>(dO, s_dO, q_idx, seq_len, qkv_offset, tid_x, tid_y);
    
    if (tid_y == 0) {
        if (q_idx >= 0 && q_idx < seq_len) {
            s_D[tid_x] = __bfloat162float(D[ml_offset + q_idx]);
        } else {
            s_D[tid_x] = 0.0f;
        }
    }
    
    if (tid_y == 0) {
        if (q_idx >= 0 && q_idx < seq_len) {
            s_row_sum[tid_x] = __bfloat162float(L[ml_offset + q_idx]);
        } else {
            s_row_sum[tid_x] = 2000.0f;
        }
    }
    
    __syncthreads();
    
    const int reduced_k_block_size = BLOCK_SIZE_X - 4*pad_k;
    int actual_q_pos = q_start + BLOCK_SIZE_X;
    int k_end = causal ? (actual_q_pos >= 0 ? actual_q_pos + 1 : 0) : seq_len;
    k_end = min(k_end, seq_len);
    
    for (int k_block = 0; k_block < (k_end + reduced_k_block_size - 1) / reduced_k_block_size; k_block++) {
        const int k_start = k_block * reduced_k_block_size - 2*pad_k;
        const int k_idx = k_start + tid_x;
        
        // Load Q for this iteration (Q shares memory with P, so load each iteration)
        load_matrix<BLOCK_SIZE_Y, D_HEAD>(Q, s_Q, q_idx, seq_len, qkv_offset, tid_x, tid_y);
        
        __syncthreads(); // TODO: erase?
        
        load_matrix<BLOCK_SIZE_Y, D_HEAD>(K, s_Kv, k_idx, seq_len, qkv_offset, tid_x, tid_y);
        
        __syncthreads();
        
        // Compute Q * K^T using common WMMA function
        wmma_multiply_bf16_to_float<false, false>(
            s_Q,                    // Matrix A (Q)
            s_Kv,                   // Matrix B (K, column-major for transpose)
            s_QK,                   // Output matrix C (Q*K^T)
            D_HEAD + 4,             // Leading dimension of A
            D_HEAD + 4,             // Leading dimension of B
            BLOCK_SIZE_X + 2,       // Leading dimension of C
            BLOCK_SIZE_X,           // M dimension
            BLOCK_SIZE_X,           // N dimension
            D_HEAD,                 // K dimension
            scale,                  // Scaling factor
            warp_id,                // Warp ID
            blockDim.y*blockDim.x/32              // Number of warps
        );
        
        __syncthreads();

        // Apply 2D convolution and compute P = exp(conv(QK^T) - L) using common function
        const int total_groups = (BLOCK_SIZE_X * BLOCK_SIZE_Y) / GROUP_SIZE;
        const int rows_per_group = (BLOCK_SIZE_X + total_groups - 1) / total_groups;
        
        for (int row_offset = 0; row_offset < rows_per_group; row_offset++) {
            const int q_row = row_group_id * rows_per_group + row_offset;
            if (q_row < BLOCK_SIZE_X) {
                const int start_k = thread_in_group * elements_per_thread;
                
                float rP[elements_per_thread];
                
                compute_2d_convolution<KX, KY, BLOCK_SIZE_X, elements_per_thread, GROUP_SIZE>(
                    s_QK,                        // Input: QK matrix
                    s_conv_kernel,               // Convolution kernel
                    BLOCK_SIZE_X + 2,           // input_lda (QK matrix leading dimension)
                    q_row,                      // Current row being processed
                    start_k,                    // Starting k position for this thread
                    row_group,                  // Row group for shuffles
                    thread_in_group,            // Thread index within group
                    pad_q,                      // Query padding left
                    0,                          // Query padding right (ignored)
                    pad_k,                      // Key padding
                    q_start,                    // q_start for global position calculation
                    k_start,                    // Block start position for keys
                    seq_len,                    // Sequence length
                    causal,                     // Whether to apply causal masking
                    -FLT_MAX,                   // Default value for invalid positions
                    rP                          // Output register array
                );
                
                // Handle backward pass post-processing: P = exp(conv(QK^T) - L)
                // Get L value for this row from s_row_sum (L = M + log(sum))
                float L_val = s_row_sum[q_row];
                
                #pragma unroll
                for (int i = 0; i < elements_per_thread; i++) {
                    float exp_value = __expf(rP[i] - L_val);
                    int idx = start_k + i + pad_k;
                    if (idx >= 0 && idx < BLOCK_SIZE_X) {
                        float safe_value = isnan(exp_value) ? 0.0f : exp_value;
                        s_P[q_row * BLOCK_SIZE_X + idx] = __float2bfloat16(exp_value);
                    } else {
                        int wrap_idx = (idx+BLOCK_SIZE_X)%BLOCK_SIZE_X;
                        if (wrap_idx < pad_k) {
                            s_P[q_row * BLOCK_SIZE_X + wrap_idx] = __float2bfloat16(0.0f);
                        }
                    }
                }
            }
        }

        __syncthreads(); // TODO: erase?

        // Load V tile using common function (V will overwrite K in s_Kv)
        load_matrix<BLOCK_SIZE_Y, D_HEAD>(V, s_Kv, k_idx, seq_len, qkv_offset, tid_x, tid_y);
        
        __syncthreads();
        
        // Compute dP = dO * V^T using WMMA
        // dO is [BLOCK_SIZE_X, D_HEAD], V is [BLOCK_SIZE_X, D_HEAD]
        // We need dO * V^T which is [BLOCK_SIZE_X, BLOCK_SIZE_X]
        
        // Note: s_dO now has its own separate memory space, no conflict with V
        
        // Compute dO * V^T using common WMMA function
        wmma_multiply_bf16_to_float<false, false>(
            s_dO,                   // Matrix A (dO) - in separate memory space
            s_Kv,                   // Matrix B (V, column-major for transpose)
            s_dP,                   // Output matrix C (dO*V^T)
            D_HEAD + 4,             // Leading dimension of A
            D_HEAD + 4,             // Leading dimension of B
            BLOCK_SIZE_X,           // Leading dimension of C
            BLOCK_SIZE_X,           // M dimension
            BLOCK_SIZE_X,           // N dimension
            D_HEAD,                 // K dimension
            1.0f,                   // Scaling factor (no scaling needed here)
            warp_id,                // Warp ID
            blockDim.y*blockDim.x/32              // Number of warps
        );
        
        __syncthreads();
        
        // Compute dS = P * (dP - D)
        // P is in s_P, dP is in s_dP, D is in s_D
        // Result goes into s_dP (overwrite)
        
        const int elements_per_thread_ds = (BLOCK_SIZE_X * BLOCK_SIZE_X) / (BLOCK_SIZE_X * BLOCK_SIZE_Y);
        const int ds_start_idx = tid_linear * elements_per_thread_ds;
        
        #pragma unroll
        for (int idx = 0; idx < elements_per_thread_ds; idx++) {
            int linear_idx = ds_start_idx + idx;
            if (linear_idx < BLOCK_SIZE_X * BLOCK_SIZE_X) {
                int row = linear_idx / BLOCK_SIZE_X;
                int col = linear_idx % BLOCK_SIZE_X;
                
                if (row < BLOCK_SIZE_X) {
                    float P_val = __bfloat162float(s_P[row * BLOCK_SIZE_X + col]);
                    float dP_val = s_dP[row * BLOCK_SIZE_X + col];
                    float D_val = s_D[row];
                
                    // dS = P * (dP - D) (no division by L needed since we use L = M + log(sum))
                    s_dS[row * BLOCK_SIZE_X + col] = P_val * (dP_val - D_val);
                }
            }
        }
        
        __syncthreads();
        
        for (int ky = 0; ky < KY; ky++) {
            int sh_ky = (ky+row_group_id)%KY;
            float r_dk[KX];
            #pragma unroll
            for (int kx = 0; kx < KX; kx++) {
                r_dk[kx] = 0.0f;
            }
            for (int row_offset = 0; row_offset < rows_per_group; row_offset++) {
                const int q_row = row_group_id * rows_per_group + row_offset;
                if (q_row < BLOCK_SIZE_X) {
                    const int start_k = thread_in_group * elements_per_thread;
                    
                    compute_conv_kernel_gradient_row<KX, BLOCK_SIZE_X, elements_per_thread, GROUP_SIZE>(
                        s_QK,                        // Input: scaled QK^T matrix
                        s_dS,                        // Output gradient: dS matrix
                        BLOCK_SIZE_X + 2,           // Input leading dimension
                        BLOCK_SIZE_X,               // Output leading dimension
                        q_row,                      // Current row being processed
                        start_k,                    // Starting k position for this thread
                        sh_ky,                         // Kernel row offset
                        row_group,                  // Row group for shuffles
                        thread_in_group,            // Thread index within group
                        pad_q,                      // Query padding
                        pad_k,                      // Key padding
                        q_start,                    // Starting position for query block
                        k_start,                    // Starting position for key block
                        seq_len,                    // Sequence length
                        r_dk,                       // Output: gradient for kernel row
                        tid_linear,                 // Thread index
                        causal                     // Whether to apply causal masking
                    );

                }
            }
                    
            #pragma unroll
            for (int kx = 0; kx < KX; kx++) {
                float group_sum = r_dk[kx];
                #pragma unroll
                for (int offset = 4; offset >= 1; offset /= 2) {
                    float other_sum = row_group.shfl_down(group_sum, offset);
                    group_sum += other_sum;
                }
                
                if (thread_in_group == 0) {
                    int kernel_idx = sh_ky * KX + kx;
                    atomicAdd(&s_dconv_kernel_accum[kernel_idx], group_sum);
                }
            }
        }
        
        __syncthreads();
        
        // Apply transposed convolution to get d(QK^T) using warp shuffles
        // This is the inverse of the forward convolution
        // We need to distribute dS back through the convolution pattern
        
        // Allocate space for d(QK^T) - reuse s_P space
        // s_dQK already declared at the beginning
        
        //// Initialize d(QK^T) to zero
        //#pragma unroll
        //for (int idx = 0; idx < elements_per_thread_ds; idx++) {
        //    int linear_idx = ds_start_idx + idx;
        //    if (linear_idx < BLOCK_SIZE_X * BLOCK_SIZE_X) {
        //        s_dQK[linear_idx] = 0.0f;
        //    }
        //}
        //__syncthreads();
        
        // Apply transposed convolution using common compute_2d_convolution with flip_kernel=true
        for (int row_offset = 0; row_offset < rows_per_group; row_offset++) {
            const int q_row = row_group_id * rows_per_group + row_offset;
            
            if (q_row < BLOCK_SIZE_X) {
                const int start_k = thread_in_group * elements_per_thread;
                
                float rQK[elements_per_thread];
                
                compute_2d_convolution<KX, KY, BLOCK_SIZE_X, elements_per_thread, GROUP_SIZE, true>( // FLIP_KERNEL = true
                    s_dP,                        // Input: dS matrix
                    s_conv_kernel,               // Convolution kernel (will be flipped)
                    BLOCK_SIZE_X,                // input_lda (dS matrix leading dimension)
                    q_row,                      // Current row being processed
                    start_k,                    // Starting k position for this thread
                    row_group,                  // Row group for shuffles
                    thread_in_group,            // Thread index within group
                    0,                          // pad_q_left (no padding for transposed conv)
                    pad_q,                          // pad_q_right (no padding for transposed conv)
                    0,                          // pad_k (no padding for transposed conv)
                    0,                          // q_start (not used for transposed conv)
                    0,                          // k_start (not used for transposed conv)
                    BLOCK_SIZE_X,               // seq_len (use block size)
                    false,                      // causal (no causal masking for transposed conv)
                    0.0f,                       // Default value for invalid positions (no masking needed)
                    rQK                         // Output register array
                );
                
                

                #pragma unroll
                for (int i = 0; i < elements_per_thread; i++) {
                    int global_pos = start_k + i + pad_k;
                    if (global_pos < BLOCK_SIZE_X && q_row < BLOCK_SIZE_X) {
                        int q_pos = q_start + q_row;
                        int k_pos = global_pos + k_start;
                        bool valid = 
                            (q_pos >= 0) &&
                            (q_pos < seq_len) &&
                            (k_pos >= 0) &&
                            (k_pos < seq_len) &&
                            (q_row >= pad_q) &&
                            (q_row < BLOCK_SIZE_X-pad_q) &&
                            (global_pos < BLOCK_SIZE_X - 2*pad_k) &&
                            (global_pos >= 2*pad_k);

                        if (causal) {
                            valid = valid && (k_pos <= q_pos);
                        }

                        if (valid) {
                            s_dQK[q_row * BLOCK_SIZE_X + global_pos] = __float2bfloat16(rQK[i]);
                        } else {
                            s_dQK[q_row * BLOCK_SIZE_X + global_pos] = __float2bfloat16(0.0f);
                        }
                    }
                }
            }
        }

        
        __syncthreads();
        
        
        // Load existing dQ from global memory into accumulator (dQ_accum shares space with QK)
        if (k_block == 0) {
            // First iteration: initialize accumulator to zero
            initialize_shared_memory<BLOCK_SIZE_X, BLOCK_SIZE_Y, D_HEAD>(
                nullptr, nullptr, nullptr, s_dQ_accum, tid_x, tid_y, tid_linear, 0.0f, 0.0f, 0.0f
            );
        } else {
            // Subsequent iterations: load existing dQ values from global memory
            if (q_idx >= 0 && q_idx < seq_len && tid_x >= pad_q && tid_x < BLOCK_SIZE_X-pad_q) {
                float4* dQ_vec = (float4*)&dQ[qkv_offset + q_idx * D_HEAD];
                float4* s_dQ_vec = (float4*)&s_dQ_accum[tid_x * (D_HEAD + 4)];
                
                const int vec_elements_per_thread = (D_HEAD / 4) / BLOCK_SIZE_Y;
                const int vec_start = tid_y * vec_elements_per_thread;
                
                #pragma unroll
                for (int d = 0; d < vec_elements_per_thread; d++) {
                    s_dQ_vec[vec_start + d] = dQ_vec[vec_start + d];
                }
            }
        }
        
        __syncthreads();
        
        load_matrix<BLOCK_SIZE_Y, D_HEAD>(K, s_Kv, k_idx, seq_len, qkv_offset, tid_x, tid_y);
        
        __syncthreads();
        
        wmma_multiply_bf16_to_float<true, true>(
            s_dQK,                  // Matrix A (d(QK^T))
            s_Kv,                   // Matrix B (K, row-major)
            s_dQ_accum,             // Output matrix C (dQ accumulator, load existing values)
            BLOCK_SIZE_X,           // Leading dimension of A
            D_HEAD + 4,             // Leading dimension of B
            D_HEAD + 4,             // Leading dimension of C
            BLOCK_SIZE_X,           // M dimension
            D_HEAD,                 // N dimension
            BLOCK_SIZE_X,           // K dimension
            1.0f,                   // Scaling factor (no scaling needed here)
            warp_id,                // Warp ID
            blockDim.y*blockDim.x/32
        );
        
        __syncthreads();
        
        if (q_idx >= 0 && q_idx < seq_len && tid_x >= pad_q && tid_x < BLOCK_SIZE_X-pad_q) {
            float4* dQ_vec = (float4*)&dQ[qkv_offset + q_idx * D_HEAD];
            float4* s_dQ_vec = (float4*)&s_dQ_accum[tid_x * (D_HEAD + 4)];
            
            const int vec_elements_per_thread = (D_HEAD / 4) / BLOCK_SIZE_Y;
            const int vec_start = tid_y * vec_elements_per_thread;
            
            #pragma unroll
            for (int d = 0; d < vec_elements_per_thread; d++) {
                // Write without scaling - scale will be applied at the end
                dQ_vec[vec_start + d] = s_dQ_vec[vec_start + d];
            }
        }
    }
    
    __syncthreads();
    
    // Apply scale multiplication at the end - read accumulated dQ, multiply by scale, and write back
    if (q_idx >= 0 && q_idx < seq_len && tid_x >= pad_q && tid_x < BLOCK_SIZE_X-pad_q) {
        float4* dQ_vec = (float4*)&dQ[qkv_offset + q_idx * D_HEAD];
        
        const int vec_elements_per_thread = (D_HEAD / 4) / BLOCK_SIZE_Y;
        const int vec_start = tid_y * vec_elements_per_thread;
        
        #pragma unroll
        for (int d = 0; d < vec_elements_per_thread; d++) {
            float4 dq_val = dQ_vec[vec_start + d];
            // Apply scale multiplication only at the end
            dq_val.x *= scale;
            dq_val.y *= scale;
            dq_val.z *= scale;
            dq_val.w *= scale;
            dQ_vec[vec_start + d] = dq_val;
        }
    }
    
    // Write accumulated convolution kernel gradient to global memory
    // Only one block per head should write the gradient to avoid race conditions
    for (int i = tid_linear; i < kernel_size; i += BLOCK_SIZE_X * BLOCK_SIZE_Y) {
        // Use atomic add to accumulate gradients from all batches
        atomicAdd(&dconv_kernel[conv_kernel_offset + i], s_dconv_kernel_accum[i]);
    }
}

// Template wrapper function to launch the backward kernel
template<int KX, int KY, int D_HEAD>
void launch_dq_backward_kernel(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor dO,
    torch::Tensor D,
    torch::Tensor L,
    torch::Tensor conv_kernel,
    torch::Tensor dQ,
    torch::Tensor dconv_kernel,
    float scale,
    int pad_q,
    int pad_k,
    bool causal
) {
    const int batch_size = Q.size(0);
    const int num_heads = Q.size(1);
    const int seq_len = Q.size(2);

    
    const int BLOCK_SIZE_X = 48;  // TODO: make it flexible
    const int BLOCK_SIZE_Y = 4;
    
    const int reduced_block_size = BLOCK_SIZE_X - 2*pad_q;
    const int num_blocks_q = (seq_len + reduced_block_size - 1) / reduced_block_size;
    
    dim3 grid(num_blocks_q, num_heads, batch_size);
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    
    // Shared memory sizes
    const int s_QK_size = (BLOCK_SIZE_X + 2) * BLOCK_SIZE_X * sizeof(float);
    const int s_dP_size = BLOCK_SIZE_X * BLOCK_SIZE_X * sizeof(float);
    const int s_D_size = BLOCK_SIZE_X * sizeof(float);
    const int s_dQ_accum_size = BLOCK_SIZE_X * (D_HEAD + 4) * sizeof(float);
    const int s_row_sum_size = BLOCK_SIZE_X * sizeof(float);
    const int s_dconv_kernel_accum_size = KX * KY * sizeof(float);
    
    // Bfloat16 arrays
    const int s_Kv_size = BLOCK_SIZE_X * (D_HEAD + 4) * sizeof(__nv_bfloat16);  // for K and V
    const int s_dO_size = BLOCK_SIZE_X * (D_HEAD + 4) * sizeof(__nv_bfloat16);  // separate space for dO
    const int s_Q_size = BLOCK_SIZE_X * (D_HEAD + 4) * sizeof(__nv_bfloat16);   // shared with P
    const int s_conv_kernel_size = KX * KY * sizeof(__nv_bfloat16);
    
    // Total size: separate arrays + shared arrays (counted once each)
    const int shared_mem_size = max(s_QK_size, s_dQ_accum_size) + s_dP_size + s_D_size + s_row_sum_size + s_dconv_kernel_accum_size +
                               s_Kv_size + s_dO_size + s_Q_size + s_conv_kernel_size;

    //printf("shared_mem_size: %d\n", shared_mem_size);
    //printf("s_QK_size: %d\n", s_QK_size);
    //printf("s_dQ_accum_size: %d\n", s_dQ_accum_size);
    //printf("s_dP_size: %d\n", s_dP_size);
    //printf("s_D_size: %d\n", s_D_size);
    //printf("s_row_sum_size: %d\n", s_row_sum_size);
    //printf("s_dconv_kernel_accum_size: %d\n", s_dconv_kernel_accum_size);
    //printf("s_Kv_size: %d\n", s_Kv_size);
    //printf("s_dO_size: %d\n", s_dO_size);
    //printf("s_Q_size: %d\n", s_Q_size);
    //printf("s_conv_kernel_size: %d\n", s_conv_kernel_size);
    //printf("\n\n");
    
    dq_kernel<BLOCK_SIZE_X, BLOCK_SIZE_Y, KX, KY, D_HEAD><<<grid, block, shared_mem_size>>>(
        reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(V.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(dO.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(D.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(L.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(conv_kernel.data_ptr()),
        reinterpret_cast<float*>(dQ.data_ptr()),
        reinterpret_cast<float*>(dconv_kernel.data_ptr()),
        batch_size,
        num_heads,
        seq_len,
        scale,
        pad_q,
        pad_k,
        causal
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));
}