#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;


/**
 * Inverts a convolution kernel in shared memory for transposed convolution.
 * This flips the kernel both horizontally and vertically.
 * 
 * @param s_kernel_src: Source kernel [KY, KX]
 * @param s_kernel_dst: Destination inverted kernel [KY, KX]  
 * @param tid_linear: Linear thread ID for collaborative work
 * @param total_threads: Total number of threads available
 */
template<int KX, int KY>
__device__ void invert_kernel(
    const __nv_bfloat16* s_kernel_src,
    __nv_bfloat16* s_kernel_dst,
    int tid_linear,
    int total_threads
) {
    const int kernel_size = KX * KY;
    
    // Each thread inverts multiple elements if needed
    for (int i = tid_linear; i < kernel_size; i += total_threads) {
        int src_y = i / KX;
        int src_x = i % KX;
        
        // Flip both dimensions
        int dst_y = KY - 1 - src_y;
        int dst_x = KX - 1 - src_x;
        
        s_kernel_dst[dst_y * KX + dst_x] = s_kernel_src[src_y * KX + src_x];
    }
}

template<int KX, int KY, int BLOCK_SIZE_X, int ELEMENTS_PER_THREAD = 4, int GROUP_SIZE = 8, bool FLIP_KERNEL = false>
__device__ void compute_2d_convolution(
    const float* s_input,           // Input matrix
    const __nv_bfloat16* s_conv_kernel,  // Convolution kernel [KY, KX]
    int input_lda,                  // Leading dimension of input matrix
    int q_row,                      // Current row being processed
    int start_k,                    // Starting k position for this thread
    thread_block_tile<GROUP_SIZE> row_group, // Row group for shuffles
    int thread_in_group,            // Thread index within group
    int pad_q_left,                 // Query padding left
    int pad_q_right,                // Query padding right
    int pad_k,                      // Key padding (symmetric) 
    int q_start,                    // Block start position for queries
    int k_start,                    // Block start position for keys
    int seq_len,                    // Sequence length
    bool causal,                    // Whether to apply causal masking
    float invalid_value,            // Default value for invalid positions
    float r0[ELEMENTS_PER_THREAD]   // Output register array
) {

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int global_pos = start_k + i;
        bool valid_pos = (global_pos < BLOCK_SIZE_X-2*pad_k && global_pos >= 0 && 
                         q_row >= pad_q_left && q_row < BLOCK_SIZE_X - pad_q_right && 
                         global_pos + k_start + pad_k >= 0 && 
                         global_pos + k_start + pad_k < seq_len);
        
        if (causal && valid_pos) {
            int actual_q_pos = q_start + q_row;
            int actual_k_pos = global_pos + k_start + pad_k;
            
            if (actual_k_pos > actual_q_pos) {
                r0[i] = invalid_value;  // Causal mask
            } else {
                r0[i] = 0.0f;
            }
        } else {
            r0[i] = valid_pos ? 0.0f : invalid_value;
        }
    }
    
    #pragma unroll
    for (int ky = 0; ky < KY; ky++) {
        float K[KX];
        #pragma unroll
        for (int kx = 0; kx < KX; kx++) {
            if (FLIP_KERNEL) {
                K[kx] = __bfloat162float(s_conv_kernel[(KY - 1 - ky) * KX + (KX - 1 - kx)]);
            } else {
                K[kx] = __bfloat162float(s_conv_kernel[ky * KX + kx]);
            }
        }
        
        int src_q = q_row + ky - pad_q_left;
        
        float I[ELEMENTS_PER_THREAD];
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            int k_col = start_k + i;
            bool valid = (src_q >= 0 && src_q < BLOCK_SIZE_X && 
                         k_col >= 0 && k_col < BLOCK_SIZE_X &&
                         k_start + k_col >= 0 && k_start + k_col < seq_len);
            if (causal) {
                int actual_q_pos = q_start + src_q;
                int actual_k_pos = k_start + k_col;
                valid = valid && actual_k_pos <= actual_q_pos;
            }
            
            if (valid) {
                I[i] = s_input[src_q * input_lda + k_col];
            } else {
                I[i] = 0.0f;
            }
        }
        
        int idx = 0;
        #pragma unroll
        for (int kx = 0; kx < KX; kx++) {
            #pragma unroll
            for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
                int reg_idx = (i + idx) % ELEMENTS_PER_THREAD;
                float qk_val = I[reg_idx];
                r0[i] += K[kx] * qk_val;
            }
            
            // Shuffle elements (except on last iteration)
            if (kx < KX - 1) {
                float shuffle_val = I[idx % ELEMENTS_PER_THREAD];
                float received_val = row_group.shfl_down(shuffle_val, 1);
                I[idx % ELEMENTS_PER_THREAD] = received_val;
                idx++;
            }
        }
    }
    // Convolution computation complete - results are in r0[] register array
}

/**
 * Compute gradient for one row of the convolution kernel. (Used in the backward pass)
 * This function computes: dconv_kernel[ky, kx] += input[q+ky, k+kx] * output_grad[q, k]
 * for a specific kernel row (ky_offset).
 */
template<int KX, int BLOCK_SIZE_X, int ELEMENTS_PER_THREAD = 4, int GROUP_SIZE = 8>
__device__ void compute_conv_kernel_gradient_row(
    const float* s_input,           // Input matrix (scaled QK^T)
    const float* s_output_grad,     // Output gradient matrix (dS)
    int input_lda,                  // Leading dimension of input
    int output_lda,                 // Leading dimension of output grad
    int q_row,                      // Current row being processed
    int start_k,                    // Starting k position for this thread
    int ky_offset,                  // Kernel row offset (0 to KY-1)
    thread_block_tile<GROUP_SIZE> row_group, // Row group for shuffles
    int thread_in_group,            // Thread index within group
    int pad_q,                      // Query padding
    int pad_k,                      // Key padding
    int q_start,                    // Starting position for query block
    int k_start,                    // Starting position for key block
    int seq_len,                    // Sequence length
    float r_dk[KX],                  // Output: gradient for one kernel row
    int tid_linear,                  // Thread index
    bool causal
) {
    int input_row = q_row + ky_offset-pad_q;
    
    bool input_row_valid = (input_row >= 0 && input_row < BLOCK_SIZE_X);
    bool output_row_valid = (q_row >= 2*pad_q && q_row < BLOCK_SIZE_X);
    
    if (!input_row_valid || !output_row_valid) {
        return; 
    }
    
    float I[ELEMENTS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int k_col = start_k + i;
        bool valid = (k_col >= 0 && k_col < input_lda && 
                     k_start + k_col >= 0 && k_start + k_col < seq_len);
        if (causal) {
            int actual_q_pos = q_start + input_row;
            int actual_k_pos = k_start + k_col;
            valid = valid && actual_k_pos <= actual_q_pos;
        }
        
        if (valid) {
            I[i] = s_input[input_row * input_lda + k_col];
        } else {
            I[i] = 0.0f;
        }
    }
    
    float O[ELEMENTS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int k_col = start_k + i + pad_k; 
        bool valid = (k_col >= 0 && k_col < output_lda);
        
        if (valid) {
            O[i] = s_output_grad[q_row * output_lda + k_col];
        } else {
            O[i] = 0.0f;
        }
    }
    
    #pragma unroll
    for (int kx = 0; kx < KX; kx++) {

        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            // We need O[i+kx], use warp shuffles to get values from neighboring threads
            int block_k_pos = start_k + i + kx;
            bool valid = (block_k_pos >= 2*pad_k && block_k_pos < BLOCK_SIZE_X - 2*pad_k);
            
            int o_idx = (i+kx) % ELEMENTS_PER_THREAD;
            r_dk[kx] += valid ? O[i] * I[o_idx] : 0.0f;
        }

        // Shift the elements of I to the right
        if (kx < KX - 1) {
            int idx = kx%ELEMENTS_PER_THREAD;
            float shuffle_val = I[idx];
            float received_val = row_group.shfl_down(shuffle_val, 1);
            if (thread_in_group == GROUP_SIZE - 1) {
                received_val = 0.0f;
            }
            I[idx] = received_val;
        }
    }
    
}

/**
 * Inference-optimized convolution function that computes only the last row of output.
 * Each group processes a subset of the KY dimension instead of looping over it.
 * Results are atomically accumulated into the provided s_conv buffer.
 */
template<int KX, int KY, int BLOCK_SIZE_X, int ELEMENTS_PER_THREAD = 4, int GROUP_SIZE = 8>
__device__ void compute_inference_convolution_last_row(
    const float* s_input,
    const __nv_bfloat16* s_conv_kernel,
    int input_lda,
    int q_row_last,
    int start_k,
    thread_block block,
    int thread_in_group,
    int pad_q_left,
    int pad_q_right,
    int pad_k,
    int q_start,
    int k_start,
    int seq_len,
    float invalid_value,
    float* s_conv,
    float r0[ELEMENTS_PER_THREAD]
) {
    const int tid_linear = threadIdx.y * blockDim.x + threadIdx.x;
    const int num_groups = blockDim.x * blockDim.y / GROUP_SIZE;
    const int group_id = tid_linear / GROUP_SIZE;
    
    const int ky_per_group = (KY + num_groups - 1) / num_groups;
    const int ky_start = group_id * ky_per_group;
    const int ky_end = min(ky_start + ky_per_group, KY);
    
    thread_block_tile<32> warp = tiled_partition<32>(block);
    thread_block_tile<GROUP_SIZE> row_group = tiled_partition<GROUP_SIZE>(warp);
    
    for (int idx = tid_linear; idx < BLOCK_SIZE_X; idx += blockDim.x * blockDim.y) {
        s_conv[idx] = 0.0f;
    }

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int global_pos = start_k + i;
        bool valid_pos = (global_pos < BLOCK_SIZE_X - 2*pad_k && global_pos >= 0 && 
                         q_row_last >= pad_q_left && q_row_last < BLOCK_SIZE_X - pad_q_right && 
                         global_pos + k_start + pad_k >= 0 && 
                         global_pos + k_start + pad_k < seq_len);
        
        r0[i] = valid_pos ? 0.0f : invalid_value;
    }
    
    for (int ky = ky_start; ky < ky_end; ky++) {
        float K[KX];
        #pragma unroll
        for (int kx = 0; kx < KX; kx++) {
            K[kx] = __bfloat162float(s_conv_kernel[ky * KX + kx]);
        }
        
        int src_q = q_row_last + ky - pad_q_left;
        
        float I[ELEMENTS_PER_THREAD];
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            int k_col = start_k + i;
            bool valid = (src_q >= 0 && src_q < BLOCK_SIZE_X && 
                         k_col >= 0 && k_col < BLOCK_SIZE_X &&
                         k_start + k_col >= 0 && k_start + k_col < seq_len);
            
            if (valid) {
                I[i] = s_input[src_q * input_lda + k_col];
            } else {
                I[i] = 0.0f;
            }
        }
        
        int idx = 0;
        #pragma unroll
        for (int kx = 0; kx < KX; kx++) {
            #pragma unroll
            for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
                int reg_idx = (i + idx) % ELEMENTS_PER_THREAD;
                float qk_val = I[reg_idx];
                r0[i] += K[kx] * qk_val;
            }
            
            if (kx < KX - 1) {
                float shuffle_val = I[idx % ELEMENTS_PER_THREAD];
                float received_val = row_group.shfl_down(shuffle_val, 1);
                I[idx % ELEMENTS_PER_THREAD] = received_val;
                idx++;
            }
        }
    }
    
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = start_k + i + pad_k;
        if (idx >= 0 && idx < BLOCK_SIZE_X) {
            atomicAdd(&s_conv[idx], r0[i]);
        } else {
            int pidx = (idx + BLOCK_SIZE_X) % BLOCK_SIZE_X;
            atomicAdd(&s_conv[pidx], r0[i]);
        }
    }
}