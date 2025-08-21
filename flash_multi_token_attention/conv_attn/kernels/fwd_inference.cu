#include <cuda_runtime.h>
#include <float.h>
#include <cstdio>
#include <cuda_bf16.h>
#include <stdint.h>
#include <mma.h>
#include <cooperative_groups.h>
#include "../../common/wmma_utils.cuh"
#include "../../common/conv_utils.cuh"
#include "../../common/kernels/common.h"
#include "../../common/print_matrix.cuh"

using namespace nvcuda;
using namespace cooperative_groups;

// Inference kernel with K-dimension splitting
// Processes last 16 query tokens but only writes output for the last one
// Each thread block handles a portion of the K dimension
template<int BLOCK_SIZE_Q, int BLOCK_SIZE_K, int D_HEAD, int KX = 3, int KY = 3>
__global__ void fwd_inference_kernel(
    const __nv_bfloat16* __restrict__ Q,      // [batch, heads, 16, d_head] - last 16 query tokens
    const __nv_bfloat16* __restrict__ K,      // [batch, heads, seq_len, d_head]
    const __nv_bfloat16* __restrict__ V,      // [batch, heads, seq_len, d_head]
    const __nv_bfloat16* __restrict__ conv_kernel,    // [heads, kx, ky]
    float* __restrict__ O,                    // [batch, heads, k_splits, d_head]
    float* __restrict__ M,                    // [batch, heads, k_splits] - max values
    float* __restrict__ L,                    // [batch, heads, k_splits] - sum of softmax
    const int batch_size,
    const int num_heads,
    const int seq_len_k,
    const int k_splits,
    const int num_k_blocks,
    const float scale,
    const int pad_q,
    const int pad_k,
    const bool causal
) {
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    const int tid_linear = tid_y * BLOCK_SIZE_K + tid_x;
    const int num_threads = blockDim.y * blockDim.x;
    
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int k_split_idx = blockIdx.x;
    const int reduced_k_block_size = BLOCK_SIZE_K - 2*pad_k;
    const int k_start_global = k_split_idx * reduced_k_block_size*num_k_blocks;
    
    
    const int q_offset = (batch_idx * num_heads + head_idx) * 16 * D_HEAD;  // Only 16 query tokens
    const int kv_offset = (batch_idx * num_heads + head_idx) * seq_len_k * D_HEAD;
    const int conv_kernel_offset = head_idx * KX * KY;
    const int output_offset = (batch_idx * num_heads + head_idx) * k_splits * D_HEAD + k_split_idx * D_HEAD;
    const int ml_offset = (batch_idx * num_heads + head_idx) * k_splits + k_split_idx;
    
    // Shared memory allocations
    extern __shared__ float shared_mem[];
    
    constexpr int S_Q_ELEMENTS = BLOCK_SIZE_Q * (D_HEAD + 4);
    constexpr int S_CONV_ELEMENTS = BLOCK_SIZE_K;  
    constexpr int S_Q_CONV_MAX_ELEMENTS = (S_Q_ELEMENTS > S_CONV_ELEMENTS) ? S_Q_ELEMENTS : S_CONV_ELEMENTS;

    __nv_bfloat16* s_Q = (__nv_bfloat16*)shared_mem;
    __nv_bfloat16* s_Kv = s_Q + S_Q_CONV_MAX_ELEMENTS;                            // [BLOCK_SIZE_K, D_HEAD + 4] - shared for K and V
    float* s_QK = (float*)(s_Kv + BLOCK_SIZE_K * (D_HEAD + 4));                          // [BLOCK_SIZE_Q, BLOCK_SIZE_K + 2] as float
    float* s_conv = (float*)(s_QK + BLOCK_SIZE_Q * (BLOCK_SIZE_K + 2));                                       // shares memory with s_Q (after Q@K computation)
    float* s_O = (float*)(s_conv + BLOCK_SIZE_K);
    float* s_row_max = (float*)(s_O + D_HEAD);                 // single element for inference
    float* s_row_exp_diff = s_row_max + 1;                                          // single element for inference
    float* s_row_sum = s_row_exp_diff + 1;                                          // single element for inference
    __nv_bfloat16* s_conv_kernel = (__nv_bfloat16*)(s_row_sum + 1);                // [KX, KY] - moved to last

    if (tid_linear == 0) {
        s_row_max[0] = -FLT_MAX;
        s_row_exp_diff[0] = 1.0f;
        s_row_sum[0] = 0.0f;  
    }
    for (int i = tid_linear; i < D_HEAD; i += num_threads) {
        s_O[i] = 0.0f;
    }


    // Warp ID for WMMA instructions
    const int warp_id = tid_linear / 32;
    
    // Cooperative groups setup
    constexpr int GROUP_SIZE = 8;
    constexpr int elements_per_thread = BLOCK_SIZE_K / GROUP_SIZE;
    
    thread_block block = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(block);  
    thread_block_tile<GROUP_SIZE> row_group = tiled_partition<GROUP_SIZE>(warp);
    
    const int row_group_id = tid_linear / GROUP_SIZE;
    const int thread_in_group = tid_linear % GROUP_SIZE;
    
    const int kernel_size = KX * KY;
    for (int i = tid_linear; i < kernel_size; i += num_threads) {
        s_conv_kernel[i] = conv_kernel[conv_kernel_offset + i];
    }
    
    const int total_threads = num_threads;  // Actual threads in the block
    load_matrix_general_vectorized<D_HEAD>(Q, s_Q, 0, 16, 16, q_offset, total_threads, tid_linear);
    __syncthreads();
    
    for (int k_tile = 0; k_tile < num_k_blocks; k_tile++) {
        const int k_start_local = k_tile * reduced_k_block_size - pad_k;
        const int k_start_absolute = k_start_global + k_start_local;
        
        if (k_start_absolute + pad_k>= seq_len_k) break;
        
        load_matrix_general_vectorized<D_HEAD>(K, s_Kv, k_start_absolute, BLOCK_SIZE_K, seq_len_k, kv_offset, total_threads, tid_linear);
        __syncthreads();
        
        
        
        wmma_multiply_bf16_to_float<false, false>(
            s_Q,
            s_Kv, 
            s_QK,
            D_HEAD + 4,  // lda
            D_HEAD + 4,  // ldb
            BLOCK_SIZE_K + 2,  // ldc
            BLOCK_SIZE_Q,  // M
            BLOCK_SIZE_K,  // N
            D_HEAD,  // K
            scale,
            warp_id,
            (blockDim.y*blockDim.x+31)/32
        );
        
        __syncthreads();
        
        
        const int last_q_row = 15;
        const int start_k = thread_in_group * elements_per_thread;
        
        
        float rO[elements_per_thread];
        
        compute_inference_convolution_last_row<KX, KY, BLOCK_SIZE_K, elements_per_thread, GROUP_SIZE>(
            s_QK,
            s_conv_kernel,
            BLOCK_SIZE_K + 2,
            last_q_row,
            start_k,
            block,
            thread_in_group,
            pad_q,
            0,
            pad_k,
            0,  // q_start is 0 since we're processing 16 tokens starting from 0
            k_start_absolute,
            seq_len_k,
            -FLT_MAX,
            s_conv,
            rO
        );
        // Convolution results are now accumulated in s_conv by atomic writes
        __syncthreads();
        
        float tile_max = -FLT_MAX;
        float tile_sum = 0.0f;
        
        // First pass: find tile max
        {
            const int lane_id = tid_linear & 31;
            const unsigned full_mask = 0xffffffffu;
            if (warp_id == 0) {
                float conv_val = s_conv[lane_id];

                float warp_max = conv_val;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    warp_max = fmaxf(warp_max, __shfl_down_sync(full_mask, warp_max, offset));
                }
                if (lane_id == 0) {
                    tile_max = warp_max;
                }
                // Broadcast tile_max to all threads in warp 0
                tile_max = __shfl_sync(full_mask, tile_max, 0);
            }
        }
        __syncthreads();
        
        // Broadcast tile_max to all threads in the block via shared memory
        if (warp_id == 0 && (tid_linear & 31) == 0) {
            s_row_exp_diff[0] = tile_max;  // Reuse shared memory slot
        }
        __syncthreads();
        tile_max = s_row_exp_diff[0];
        
        // Update global max and compute correction factors
        float global_max_old = s_row_max[0];
        float global_max_new = fmaxf(global_max_old, tile_max);
        float correction_factor = (global_max_old == -FLT_MAX) ? 1.0f : expf(global_max_old - global_max_new);
        float tile_correction = expf(tile_max - global_max_new);
        
        if (tid_linear == 0) {
            s_row_max[0] = global_max_new;
        }
        // Rescale previous output if needed
        if (k_tile > 0 && correction_factor != 1.0f) {
            for (int i = tid_linear; i < D_HEAD; i += num_threads) {
                s_O[i] *= correction_factor;
            }
        }

        
        __syncthreads();
        
        // Second pass: compute tile softmax with corrected max
        {
            const int lane_id = tid_linear & 31;
            const unsigned full_mask = 0xffffffffu;
            if (warp_id == 0) {
                // Compute exp(val - global_max_new) and write back to shared memory
                float conv_val = s_conv[lane_id];
                float exp_val = expf(conv_val - global_max_new);
                s_conv[lane_id] = exp_val;

                // Reduce-sum across the warp
                float warp_sum = exp_val;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    warp_sum += __shfl_down_sync(full_mask, warp_sum, offset);
                }
                if (lane_id == 0) {
                    tile_sum = warp_sum;
                }
            }
        }

        __syncthreads();
        
        // Update global sum with correction
        if (tid_linear == 0) {
            s_row_sum[0] = s_row_sum[0] * correction_factor + tile_sum;
        }
        __syncthreads();
        
        // Load V tile into shared memory (reusing s_Kv)
        load_matrix_general_vectorized<D_HEAD>(V, s_Kv, k_start_absolute, BLOCK_SIZE_K, seq_len_k, kv_offset, total_threads, tid_linear);
        __syncthreads();
        
        

        // Compute O = conv x V without wmma
        // conv is (1, BLOCK_SIZE_K), V is (BLOCK_SIZE_K, D_HEAD), O is (1, D_HEAD)
        const int elements_per_thread_o = (D_HEAD + num_threads - 1) / num_threads;
        
        for (int elem_idx = 0; elem_idx < elements_per_thread_o; elem_idx++) {
            const int d_idx = tid_linear * elements_per_thread_o + elem_idx;
            if (d_idx < D_HEAD) {
                float accumulator = 0.0f;
                
                for (int k_idx = 0; k_idx < BLOCK_SIZE_K; k_idx++) {
                    if (k_start_absolute + k_idx < seq_len_k) {
                        float conv_val = s_conv[k_idx];
                        __nv_bfloat16 v_val = s_Kv[k_idx * (D_HEAD + 4) + d_idx];
                        accumulator += conv_val * __bfloat162float(v_val);
                    }
                }
                
                s_O[d_idx] += accumulator;
            }
        }
        __syncthreads();
    }
    __syncthreads();


    const int last_q_row = 15;
    for (int i = tid_linear; i < D_HEAD; i += num_threads) {
        O[output_offset + i] = s_O[i];
    }
    
    if (tid_linear == 0) {
        M[ml_offset] = s_row_max[0];
        L[ml_offset] = s_row_sum[0];
    }
    
}

// Kernel to combine results from different k_splits
template<int D_HEAD>
__global__ void combine_splits_kernel(
    float* __restrict__ O,     // [batch, heads, k_splits, d_head]
    float* __restrict__ M,     // [batch, heads, k_splits] - max values
    float* __restrict__ L,     // [batch, heads, k_splits] - sum of softmax
    const int batch_size,
    const int num_heads,
    const int k_splits
) {
    // Each block handles one (batch, head) pair
    const int batch_idx = blockIdx.x / num_heads;
    const int head_idx = blockIdx.x % num_heads;
    
    if (batch_idx >= batch_size) return;
    
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    
    // Calculate base offsets for this (batch, head) pair
    const int ml_base_offset = (batch_idx * num_heads + head_idx) * k_splits;
    const int output_base_offset = (batch_idx * num_heads + head_idx) * k_splits * D_HEAD;
    
    // Shared memory for current M, L values and temporary storage
    extern __shared__ float shared_combine[];
    float* s_M_values = shared_combine;  // [k_splits]
    float* s_L_values = s_M_values + k_splits;  // [k_splits]
    float* s_temp_O = s_L_values + k_splits;  // [D_HEAD] - temporary storage for output
    
    // Load all M and L values for this (batch, head) pair
    for (int split = tid; split < k_splits; split += num_threads) {
        s_M_values[split] = M[ml_base_offset + split];
        s_L_values[split] = L[ml_base_offset + split];
    }
    __syncthreads();
    
    // Initialize with first split (split 0)
    float global_M = s_M_values[0];
    float global_L = s_L_values[0];
    
    // Load first split's output
    for (int i = tid; i < D_HEAD; i += num_threads) {
        s_temp_O[i] = O[output_base_offset + i];  // First split's output
    }
    __syncthreads();
    
    // Iteratively combine with other splits
    for (int other_split = 1; other_split < k_splits; other_split++) {
        float other_M = s_M_values[other_split];
        float other_L = s_L_values[other_split];
        
        // Compute new global maximum
        float new_global_M = fmaxf(global_M, other_M);
        
        // Compute correction factors
        float current_correction = (global_M == -FLT_MAX) ? 0.0f : expf(global_M - new_global_M);
        float other_correction = (other_M == -FLT_MAX) ? 0.0f : expf(other_M - new_global_M);
        
        // Update global L
        global_L = global_L * current_correction + other_L * other_correction;
        global_M = new_global_M;
        
        // Load other split's output and combine
        const int other_output_offset = output_base_offset + other_split * D_HEAD;
        for (int i = tid; i < D_HEAD; i += num_threads) {
            float other_O = O[other_output_offset + i];
            float current_O = s_temp_O[i];
            
            // Combine outputs with proper scaling
            s_temp_O[i] = current_O * current_correction + other_O * other_correction;
        }
        __syncthreads();
    }
    
    // Write final combined results back to the first split's position
    for (int i = tid; i < D_HEAD; i += num_threads) {
        O[output_base_offset + i] = s_temp_O[i]/global_L;
    }
    
    // Write final M and L values to the first split's position
    if (tid == 0) {
        M[ml_base_offset] = global_M;
        L[ml_base_offset] = global_L;
    }
}

// Template launcher function for the inference kernel
template<int KX, int KY, int D_HEAD>
void launch_inference_kernel(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    const __nv_bfloat16* conv_kernel,
    float* O,
    float* M,
    float* L,
    int batch_size,
    int num_heads,
    int seq_len_k,
    int k_splits,
    float scale,
    int pad_q,
    int pad_k,
    bool causal,
    int num_k_blocks
) {
    const int BLOCK_SIZE_Q = 16;  // Smaller block size for Q dimension
    const int BLOCK_SIZE_K = 32;  // Different block size for K dimension
    
    
    dim3 grid(k_splits, num_heads, batch_size);
    dim3 block(BLOCK_SIZE_K, 4);  // 2D thread block
    
    // Calculate shared memory size
    const int s_Q_size = BLOCK_SIZE_Q * (D_HEAD + 4) * sizeof(__nv_bfloat16);
    const int s_conv_size = BLOCK_SIZE_K * sizeof(float);  // Only one row for inference
    const int s_Kv_size = BLOCK_SIZE_K * (D_HEAD + 4) * sizeof(__nv_bfloat16);
    const int s_QK_size = (BLOCK_SIZE_K + 2) * BLOCK_SIZE_Q * sizeof(float);
    const int s_O_size = D_HEAD * sizeof(float);
    const int s_row_max_size = sizeof(float);  // Single element for inference
    const int s_row_exp_diff_size = sizeof(float);  // Single element for inference
    const int s_row_sum_size = sizeof(float);  // Single element for inference
    const int s_conv_kernel_size = KX * KY * sizeof(__nv_bfloat16);
    const int warp_reduction_size = 32 * sizeof(float);  // For warp reduction
    
    const int shared_mem_size = s_Q_size + s_conv_size + s_Kv_size + s_QK_size + s_O_size +
                               s_row_max_size + s_row_exp_diff_size + s_row_sum_size + s_conv_kernel_size + warp_reduction_size;
    
    fwd_inference_kernel<BLOCK_SIZE_Q, BLOCK_SIZE_K, D_HEAD, KX, KY><<<grid, block, shared_mem_size>>>(
        Q, K, V, conv_kernel, O, M, L,
        batch_size, num_heads, seq_len_k, k_splits, num_k_blocks, scale, pad_q, pad_k, causal
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA inference kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }
    
    // Wait for all inference kernels to complete
    cudaDeviceSynchronize();
    
    // Launch combine splits kernel only if we have multiple splits
    if (k_splits > 1) {
        dim3 combine_grid(batch_size * num_heads);
        dim3 combine_block(256);  // Use 256 threads per block
        
        // Calculate shared memory size for combine kernel
        const int combine_shared_mem_size = (2 * k_splits + D_HEAD) * sizeof(float);
        
        combine_splits_kernel<D_HEAD><<<combine_grid, combine_block, combine_shared_mem_size>>>(
            O, M, L, batch_size, num_heads, k_splits
        );
        
        cudaError_t combine_err = cudaGetLastError();
        if (combine_err != cudaSuccess) {
            printf("CUDA combine splits kernel launch failed: %s\n", cudaGetErrorString(combine_err));
        }
    }
}