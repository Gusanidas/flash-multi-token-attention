#include <cuda_runtime.h>
#include <float.h>
#include <cstdio>
#include <cuda_bf16.h>
#include <mma.h>
#include <cooperative_groups.h>
#include "../../common/wmma_utils.cuh"
#include "../../common/conv_utils.cuh"
#include "../../common/post_process_conv_fwd.cuh"
#include "../../common/kernels/common.h"
#include "../../common/print_matrix.cuh"

using namespace nvcuda;
using namespace cooperative_groups;



// 2D thread dimension version of attention with 2D convolution
// Uses configurable BLOCK_SIZE_X x BLOCK_SIZE_Y thread block
// Each thread loads a portion of the data, and uses WMMA for matrix multiplication
template<int BLOCK_SIZE_X, int BLOCK_SIZE_Y, int D_HEAD, int KX = 3, int KY = 3>
__global__ void fwd_kernel(
    const __nv_bfloat16* __restrict__ Q,      // [batch, heads, seq_len, d_head]
    const __nv_bfloat16* __restrict__ K,      // [batch, heads, seq_len, d_head]
    const __nv_bfloat16* __restrict__ V,      // [batch, heads, seq_len, d_head]
    const __nv_bfloat16* __restrict__ conv_kernel,    // [heads, kx, ky]
    float* __restrict__ O,                    // [batch, heads, seq_len, d_head]
    float* __restrict__ M,                    // [batch, heads, seq_len] - max values
    float* __restrict__ L,                    // [batch, heads, seq_len] - sum of softmax
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
    const int reduced_block_size = BLOCK_SIZE_X - pad_q;
    const int q_start = q_block_idx * reduced_block_size-pad_q;
    const int q_idx = q_start + tid_x;
    
    
    // Calculate base offsets for this batch and head
    const int qkv_offset = (batch_idx * num_heads + head_idx) * seq_len * D_HEAD;
    const int conv_kernel_offset = head_idx * KX * KY;
    const int ml_offset = (batch_idx * num_heads + head_idx) * seq_len;
    
    // Shared memory allocations
    extern __shared__ float shared_mem[];
    
    constexpr int S_Q_ELEMENTS = BLOCK_SIZE_X * (D_HEAD + 4);
    constexpr int S_CONV_ELEMENTS = BLOCK_SIZE_X * BLOCK_SIZE_X;
    // Use C++ style max for constexpr
    constexpr int S_Q_CONV_MAX_ELEMENTS = (S_Q_ELEMENTS > S_CONV_ELEMENTS) ? S_Q_ELEMENTS : S_CONV_ELEMENTS;

    __nv_bfloat16* s_Q = (__nv_bfloat16*)shared_mem;
    __nv_bfloat16* s_Kv = s_Q + S_Q_CONV_MAX_ELEMENTS;                            // [BLOCK_SIZE_X, D_HEAD + 4] - shared for K and V
    float* s_QK = (float*)(s_Kv + BLOCK_SIZE_X * (D_HEAD + 4));                          // [BLOCK_SIZE_X + 2, BLOCK_SIZE_X] as float
    __nv_bfloat16* s_conv = (__nv_bfloat16*)(s_Q);                                       // shares memory with s_Q (after Q@K computation)
    float* s_O = (float*)(s_QK);
    float* s_row_max = (float*)(s_O + BLOCK_SIZE_X * (D_HEAD + 4));                 // [BLOCK_SIZE_X] for max values
    float* s_row_exp_diff = s_row_max + BLOCK_SIZE_X;                               // [BLOCK_SIZE_X] for exp_diff values
    float* s_row_sum = s_row_exp_diff + BLOCK_SIZE_X;                               // [BLOCK_SIZE_X] for sum values
    __nv_bfloat16* s_conv_kernel = (__nv_bfloat16*)(s_row_sum + BLOCK_SIZE_X);     // [KX, KY] - moved to last
    
    // Warp ID for WMMA instructions
    const int warp_id = tid_linear / 32;
    
    // Calculate group size based on block size for optimal memory access
    // Each thread should handle BLOCK_SIZE_X/GROUP_SIZE elements 
    constexpr int GROUP_SIZE = 16;  // TODO: this should be a template parameter
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
    }
    
    float row_max_init_val = -FLT_MAX;
    float row_sum_init_val = 1.0f;
    float row_exp_diff_init_val = 0.0f;
    // Initialize shared memory
    initialize_shared_memory<BLOCK_SIZE_X, BLOCK_SIZE_Y, D_HEAD>(
        s_row_max, s_row_exp_diff, s_row_sum, s_O, tid_x, tid_y, tid_linear, row_max_init_val, row_sum_init_val, row_exp_diff_init_val
    );
    
    //__syncthreads();
    
    const int reduced_k_block_size = BLOCK_SIZE_X - 2*pad_k;
    int actual_q_pos = q_start+BLOCK_SIZE_X;  // actual query position in sequence
    int k_end = causal ? (actual_q_pos >= 0 ? actual_q_pos + 1 : 0) : seq_len;
    
    k_end = min(k_end, seq_len);
    for (int k_block = 0; k_block < (k_end + reduced_k_block_size - 1) / reduced_k_block_size; k_block++) {
        const int k_start = k_block * reduced_k_block_size - pad_k;
        
        // Load Q for this iteration into shared memory (Q shares memory with P, so load each iteration)
        load_matrix<BLOCK_SIZE_Y, D_HEAD>(Q, s_Q, q_idx, seq_len, qkv_offset, tid_x, tid_y);
        
        //__syncthreads();
        
        const int k_idx = k_start + tid_x;
        load_matrix<BLOCK_SIZE_Y, D_HEAD>(K, s_Kv, k_idx, seq_len, qkv_offset, tid_x, tid_y);
        
        __syncthreads();
        
        wmma_multiply_bf16_to_float<false, false>(
            s_Q,
            s_Kv, 
            s_QK,
            D_HEAD + 4,  // lda ( +4 to avoid bank conflicts)
            D_HEAD + 4,  // ldb
            BLOCK_SIZE_X + 2,  // ldc
            BLOCK_SIZE_X,  // M
            BLOCK_SIZE_X,  // N
            D_HEAD,  // K
            scale,
            warp_id,
            blockDim.y*blockDim.x/32
        );
        
        __syncthreads();

        const int total_groups = (BLOCK_SIZE_X * BLOCK_SIZE_Y) / GROUP_SIZE;
        const int rows_per_group = (BLOCK_SIZE_X + total_groups - 1) / total_groups;
        
        for (int row_offset = 0; row_offset < rows_per_group; row_offset++) {
            const int q_row = row_group_id * rows_per_group + row_offset;
            
            const int start_k = thread_in_group * elements_per_thread;
            if (q_row < BLOCK_SIZE_X && (q_start + q_row) < seq_len) {
            // Each thread in the group handles BLOCK_SIZE_X/GROUP_SIZE elements
            
            // Initialize register array for convolution results
            float rO[elements_per_thread];
            
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
                q_start,                    // Block start position for queries
                k_start,                    // Block start position for keys
                seq_len,                    // Sequence length
                causal,                     // Whether to apply causal masking
                -FLT_MAX,                   // Default value for invalid positions
                rO                          // Output register array
            );
            
            post_process_conv_fwd<BLOCK_SIZE_X, D_HEAD, elements_per_thread, GROUP_SIZE>(
                rO,                     // Input convolution results
                s_conv,                 // Output shared memory
                s_row_max,              // Max values shared memory
                s_row_exp_diff,         // Exp diff shared memory
                s_row_sum,              // Sum values shared memory
                q_row,                  // Current row
                start_k,                // Starting k position
                pad_k,                  // Key padding
                row_group,              // Thread group
                thread_in_group         // Thread index in group
            );


            } else {
                #pragma unroll
                for (int i = 0; i < elements_per_thread; i++) {
                    s_conv[q_row * BLOCK_SIZE_X + start_k + i] = __float2bfloat16(0.0f);
                }
            }
             
        }
        __syncthreads();

        load_matrix<BLOCK_SIZE_Y, D_HEAD>(V, s_Kv, k_idx, seq_len, qkv_offset, tid_x, tid_y);
        
        load_matrix<BLOCK_SIZE_Y, D_HEAD>(O, s_O, q_idx, seq_len, qkv_offset, tid_x, tid_y);
        
        __syncthreads();
        
        const int elements_per_thread_o = D_HEAD / BLOCK_SIZE_Y;
        const int d_start_o = tid_y * elements_per_thread_o;
        
        float row_exp_diff = (tid_x < BLOCK_SIZE_X) ? s_row_exp_diff[tid_x] : 1.0f;
        
        #pragma unroll
        for (int d = 0; d < elements_per_thread_o; d++) {
            int idx = tid_x * (D_HEAD + 4) + d_start_o + d;
            if ((d_start_o + d) < D_HEAD) {
                s_O[idx] *= row_exp_diff;
            }
        }
        
        __syncthreads();
        
        wmma_multiply_bf16_to_float<true, true>(
            s_conv,
            s_Kv,
            s_O,
            BLOCK_SIZE_X,  // lda
            D_HEAD + 4,    // ldb
            D_HEAD + 4,    // ldc
            BLOCK_SIZE_X,  // M
            D_HEAD,        // N
            BLOCK_SIZE_X,  // K
            1.0f,          // scale
            warp_id,
            blockDim.y*blockDim.x/32
        );
        
        __syncthreads();
        
        if (q_idx >= 0 && q_idx < seq_len && tid_x >= pad_q) {
            float4* O_vec = (float4*)&O[qkv_offset + q_idx * D_HEAD];
            float4* s_O_vec = (float4*)&s_O[tid_x * (D_HEAD + 4)];
            
            const int vec_elements_per_thread = (D_HEAD / 4) / BLOCK_SIZE_Y;
            const int vec_start = tid_y * vec_elements_per_thread;
            
            #pragma unroll
            for (int d = 0; d < vec_elements_per_thread; d++) {
                O_vec[vec_start + d] = s_O_vec[vec_start + d];
            }
            
        }
    }
    
    __syncthreads();
    
    // Write output with normalization
    normalize_and_write_output<BLOCK_SIZE_X, BLOCK_SIZE_Y, D_HEAD>(
        O, M, L, s_O, s_row_max, s_row_sum, qkv_offset, ml_offset, q_idx, seq_len, pad_q, tid_x, tid_y
    );
}

// Template launcher function for the forward kernel
template<int KX, int KY, int D_HEAD>
void launch_forward_kernel(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    const __nv_bfloat16* conv_kernel,
    float* O,
    float* M,
    float* L,
    int batch_size,
    int num_heads,
    int seq_len,
    float scale,
    int pad_q,
    int pad_k,
    bool causal
) {
    const int BLOCK_SIZE_X = 32;
    const int BLOCK_SIZE_Y = 4;
    
    // Calculate grid dimensions
    const int reduced_block_size = BLOCK_SIZE_X - pad_q;
    const int num_blocks_q = (seq_len + reduced_block_size - 1) / reduced_block_size;
    
    dim3 grid(num_blocks_q, num_heads, batch_size);
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    
    // Calculate shared memory size
    const int s_Q_size = BLOCK_SIZE_X * (D_HEAD + 4) * sizeof(__nv_bfloat16);
    const int s_Kv_size = BLOCK_SIZE_X * (D_HEAD + 4) * sizeof(__nv_bfloat16);
    const int s_QK_size = (BLOCK_SIZE_X + 2) * BLOCK_SIZE_X * sizeof(float);
    const int s_O_size = BLOCK_SIZE_X * (D_HEAD + 4) * sizeof(float);
    const int s_conv_size = BLOCK_SIZE_X * BLOCK_SIZE_X * sizeof(__nv_bfloat16);
    const int s_row_max_size = BLOCK_SIZE_X * sizeof(float);
    const int s_row_exp_diff_size = BLOCK_SIZE_X * sizeof(float);
    const int s_row_sum_size = BLOCK_SIZE_X * sizeof(float);
    const int s_conv_kernel_size = KX * KY * sizeof(__nv_bfloat16);
    
    const int shared_mem_size = max(s_Q_size, s_conv_size) + s_Kv_size + max(s_QK_size, s_O_size) + s_row_max_size + s_row_exp_diff_size + s_row_sum_size + s_conv_kernel_size;
    
    fwd_kernel<BLOCK_SIZE_X, BLOCK_SIZE_Y, D_HEAD, KX, KY><<<grid, block, shared_mem_size>>>(
        Q, K, V, conv_kernel, O, M, L,
        batch_size, num_heads, seq_len, scale, pad_q, pad_k, causal
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}