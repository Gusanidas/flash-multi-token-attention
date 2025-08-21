#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

/**
 * Post-processes convolution results for the forward pass by computing:
 * 1. Maximum value across the thread group
 * 2. Exponential values with max subtraction  
 * 3. Sum of exponentials for softmax normalization
 * 4. Writing results to shared memory in bfloat16 format
 * 
 * This function handles the forward pass specific post-processing that includes
 * online softmax computation with numerical stability.
 * 
 * Template parameters:
 * @param BLOCK_SIZE_X: Block size in X dimension
 * @param D_HEAD: Head dimension (for proper stride calculation)
 * @param ELEMENTS_PER_THREAD: Number of elements per thread (compile-time constant)
 * @param GROUP_SIZE: Size of cooperative group for shuffles (compile-time constant)
 * 
 * Function parameters:
 * @param rO: Input register array containing convolution results [ELEMENTS_PER_THREAD]
 * @param s_conv: Output shared memory buffer for softmax values [BLOCK_SIZE_X, BLOCK_SIZE_X] (shares with s_Q)
 * @param s_row_max: Shared memory for maximum values [BLOCK_SIZE_X]
 * @param s_row_exp_diff: Shared memory for exponential differences [BLOCK_SIZE_X] 
 * @param s_row_sum: Shared memory for sum values [BLOCK_SIZE_X]
 * @param q_row: Current row being processed
 * @param start_k: Starting k position for this thread
 * @param pad_k: Key padding
 * @param row_group: Cooperative group for shuffle operations (GROUP_SIZE threads)
 * @param thread_in_group: Thread index within the row group
 */
template<int BLOCK_SIZE_X, int D_HEAD, int ELEMENTS_PER_THREAD = 4, int GROUP_SIZE = 8>
__device__ void post_process_conv_fwd(
    const float rO[ELEMENTS_PER_THREAD],    // Input convolution results
    __nv_bfloat16* s_conv,                  // Output shared memory [BLOCK_SIZE_X, BLOCK_SIZE_X] (shares with s_Q)
    float* s_row_max,                       // Shared memory for max values [BLOCK_SIZE_X]
    float* s_row_exp_diff,                  // Shared memory for exp differences [BLOCK_SIZE_X]
    float* s_row_sum,                       // Shared memory for sum values [BLOCK_SIZE_X]
    int q_row,                              // Current row being processed
    int start_k,                            // Starting k position for this thread
    int pad_k,                              // Key padding
    thread_block_tile<GROUP_SIZE> row_group,         // Row group for shuffles
    int thread_in_group                     // Thread index within group
) {
    float thread_max = -1000.0f;
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        thread_max = fmaxf(thread_max, rO[i]);
    }
    
    float group_max = thread_max;
    float new_max = thread_max;
    float exp_diff = 1.0f;
    #pragma unroll
    for (int offset = GROUP_SIZE/2; offset >= 1; offset /= 2) {
        float other_max = row_group.shfl_down(group_max, offset);
        group_max = fmaxf(group_max, other_max);
    }
    
    group_max = row_group.shfl(group_max, 0);
    
    if (thread_in_group == 0) {
        float old_max = s_row_max[q_row];
        new_max = fmaxf(old_max, group_max);
        s_row_max[q_row] = new_max;
        exp_diff = __expf(old_max - new_max);
        s_row_exp_diff[q_row] = exp_diff;
    }
    new_max = row_group.shfl(new_max, 0);
    exp_diff = row_group.shfl(exp_diff, 0);

    float thread_sum = 0.0f;
    
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        float exp_value = __expf(rO[i] - new_max);
        thread_sum += exp_value;
        int idx = start_k + i + pad_k;
        if (idx >= 0 && idx < BLOCK_SIZE_X) {
            float safe_value = isnan(exp_value) ? 0.0f : exp_value;
            s_conv[q_row * BLOCK_SIZE_X + idx] = __float2bfloat16(safe_value);
        }
    }

    float group_sum = thread_sum;
    #pragma unroll
    for (int offset = GROUP_SIZE/2; offset >= 1; offset /= 2) {
        float other_sum = row_group.shfl_down(group_sum, offset);
        group_sum += other_sum;
    }

    if (thread_in_group == 0) {
        float old_sum = s_row_sum[q_row];
        group_sum = old_sum * exp_diff + group_sum;
        s_row_sum[q_row] = group_sum;
    }
}