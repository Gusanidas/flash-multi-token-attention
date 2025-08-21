#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

// Common utilities 
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Helper macros for template dispatch
#define DISPATCH_KX_KY(KX, KY, ...) \
  if (KX == 3 && KY == 3) { \
    constexpr int kKX = 3; \
    constexpr int kKY = 3; \
    __VA_ARGS__; \
  } else if (KX == 5 && KY == 5) { \
    constexpr int kKX = 5; \
    constexpr int kKY = 5; \
    __VA_ARGS__; \
  } else { \
    TORCH_CHECK(false, "Unsupported kernel size: KX=", KX, ", KY=", KY); \
  }

#define DISPATCH_D_HEAD(D_HEAD, ...) \
  if (D_HEAD == 64) { \
    constexpr int kDHead = 64; \
    __VA_ARGS__; \
  } else if (D_HEAD == 128) { \
    constexpr int kDHead = 128; \
    __VA_ARGS__; \
  } else { \
    TORCH_CHECK(false, "Unsupported head dimension: D_HEAD=", D_HEAD); \
  }

// Common data structures, for vectorized loading and storing
struct __align__(8) bfloat164 {
    __nv_bfloat16 x, y, z, w;
};

template<int BLOCK_SIZE_Y, int D_HEAD>
__device__ void load_matrix(
    const __nv_bfloat16* __restrict__ global_matrix,
    __nv_bfloat16* shared_matrix,
    int matrix_idx,
    int seq_len,
    int qkv_offset,
    int tid_x,
    int tid_y
) {
    if (matrix_idx >= 0 && matrix_idx < seq_len) {
        const bfloat164* G_vec = (const bfloat164*)&global_matrix[qkv_offset + matrix_idx * D_HEAD];
        bfloat164* s_M_vec = (bfloat164*)&shared_matrix[tid_x * (D_HEAD + 4)];

        const int vec_elements_per_thread = (D_HEAD / 4) / BLOCK_SIZE_Y;
        const int vec_start = tid_y * vec_elements_per_thread;

        #pragma unroll
        for (int d = 0; d < vec_elements_per_thread; d++) {
            s_M_vec[vec_start + d] = G_vec[vec_start + d];
        }
    } else {
        const __nv_bfloat16 zero_bf16 = __float2bfloat16(0.0f);
        const int elements_per_thread = D_HEAD / BLOCK_SIZE_Y;
        const int d_start = tid_y * elements_per_thread;

        #pragma unroll
        for (int d = 0; d < elements_per_thread; d++) {
            shared_matrix[tid_x * (D_HEAD + 4) + d_start + d] = zero_bf16;
        }
    }
}

// General matrix loading function that distributes work among arbitrary number of threads
template<int D_HEAD>
__device__ void load_matrix_general(
    const __nv_bfloat16* __restrict__ global_matrix,
    __nv_bfloat16* shared_matrix,
    int matrix_start_idx,        // Starting row index in global matrix
    int num_rows,               // Number of rows to load
    int seq_len,               // Total sequence length for bounds checking
    int qkv_offset,            // Base offset in global matrix
    int total_threads,         // Total number of threads participating
    int thread_id              // This thread's ID (0 to total_threads-1)
) {
    // Calculate total elements and work distribution
    const int total_elements = num_rows * D_HEAD;
    const int elements_per_thread = (total_elements + total_threads - 1) / total_threads;
    const int start_element = thread_id * elements_per_thread;
    const int end_element = min(start_element + elements_per_thread, total_elements);
    
    // Load elements assigned to this thread
    for (int elem_idx = start_element; elem_idx < end_element; elem_idx++) {
        const int row = elem_idx / D_HEAD;
        const int col = elem_idx % D_HEAD;
        const int global_row_idx = matrix_start_idx + row;
        
        if (global_row_idx >= 0 && global_row_idx < seq_len) {
            // Load from global memory
            shared_matrix[row * (D_HEAD + 4) + col] = 
                global_matrix[qkv_offset + global_row_idx * D_HEAD + col];
        } else {
            // Zero padding for out-of-bounds
            shared_matrix[row * (D_HEAD + 4) + col] = __float2bfloat16(0.0f);
        }
    }
}

// Vectorized version for better performance when D_HEAD is divisible by 4
template<int D_HEAD>
__device__ void load_matrix_general_vectorized(
    const __nv_bfloat16* __restrict__ global_matrix,
    __nv_bfloat16* shared_matrix,
    int matrix_start_idx,        // Starting row index in global matrix
    int num_rows,               // Number of rows to load
    int seq_len,               // Total sequence length for bounds checking
    int qkv_offset,            // Base offset in global matrix
    int total_threads,         // Total number of threads participating
    int thread_id,             // This thread's ID (0 to total_threads-1)
    int row_mask_low = 0,      // Starting row for masking (default 0)
    int row_mask_high = -1     // Ending row for masking (default num_rows)
) {
    static_assert(D_HEAD % 4 == 0, "D_HEAD must be divisible by 4 for vectorized loading");
    
    // Set default row_mask_high if not specified
    const int effective_row_mask_high = (row_mask_high == -1) ? num_rows : row_mask_high;
    
    // Calculate total vector elements (each bfloat164 contains 4 elements)
    const int vec_elements_per_row = D_HEAD / 4;
    const int total_vec_elements = num_rows * vec_elements_per_row;
    const int vec_elements_per_thread = (total_vec_elements + total_threads - 1) / total_threads;
    const int start_vec_element = thread_id * vec_elements_per_thread;
    const int end_vec_element = min(start_vec_element + vec_elements_per_thread, total_vec_elements);
    
    // Load vector elements assigned to this thread
    for (int vec_idx = start_vec_element; vec_idx < end_vec_element; vec_idx++) {
        const int row = vec_idx / vec_elements_per_row;
        const int vec_col = vec_idx % vec_elements_per_row;
        const int global_row_idx = matrix_start_idx + row;
        
        const bfloat164 zero_vec = {__float2bfloat16(0.0f), __float2bfloat16(0.0f), 
                                   __float2bfloat16(0.0f), __float2bfloat16(0.0f)};
        
        // Check if row is within mask range and within sequence bounds
        if (row >= row_mask_low && row < effective_row_mask_high && 
            global_row_idx >= 0 && global_row_idx < seq_len) {
            // Load vectorized from global memory
            const bfloat164* G_vec = (const bfloat164*)&global_matrix[qkv_offset + global_row_idx * D_HEAD];
            bfloat164* s_M_vec = (bfloat164*)&shared_matrix[row * (D_HEAD + 4)];
            s_M_vec[vec_col] = G_vec[vec_col];
        } else {
            // Zero padding for out-of-bounds or masked rows
            bfloat164* s_M_vec = (bfloat164*)&shared_matrix[row * (D_HEAD + 4)];
            s_M_vec[vec_col] = zero_vec;
        }
    }
}

// load matrix function, no really general as it makes some assumptions about the thread layout
template<int BLOCK_SIZE_Y, int D_HEAD>
__device__ void load_matrix(
    const float* __restrict__ global_matrix,
    float* shared_matrix,
    int matrix_idx,
    int seq_len,
    int qkv_offset,
    int tid_x,
    int tid_y
) {
    if (matrix_idx >= 0 && matrix_idx < seq_len) {
        const float4* G_vec = (const float4*)&global_matrix[qkv_offset + matrix_idx * D_HEAD];
        float4* s_M_vec = (float4*)&shared_matrix[tid_x * (D_HEAD + 4)];

        const int vec_elements_per_thread = (D_HEAD / 4) / BLOCK_SIZE_Y;
        const int vec_start = tid_y * vec_elements_per_thread;

        #pragma unroll
        for (int d = 0; d < vec_elements_per_thread; d++) {
            s_M_vec[vec_start + d] = G_vec[vec_start + d];
        }
    } else {
        const int vec_elements_per_thread = (D_HEAD / 4) / BLOCK_SIZE_Y;
        const int vec_start = tid_y * vec_elements_per_thread;

        #pragma unroll
        for (int d = 0; d < vec_elements_per_thread; d++) {
            float4* s_M_vec = (float4*)&shared_matrix[tid_x * (D_HEAD + 4)];
            s_M_vec[vec_start + d] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
}

// Shared memory initialization function, used in fwd and dq_bwd
template<int BLOCK_SIZE_X, int BLOCK_SIZE_Y, int D_HEAD>
__device__ void initialize_shared_memory(
    float* s_row_max,
    float* s_row_exp_diff,
    float* s_row_sum,
    float* s_O,
    int tid_x,
    int tid_y,
    int tid_linear,
    float row_max_init_val,
    float row_sum_init_val,
    float row_exp_diff_init_val
) {
    if (tid_linear < BLOCK_SIZE_X) {
        s_row_max[tid_linear] = row_max_init_val;
        s_row_exp_diff[tid_linear] = row_exp_diff_init_val;
        s_row_sum[tid_linear] = row_sum_init_val;
    }

    const int floats_per_thread = D_HEAD / (BLOCK_SIZE_Y * 4);
    const int start_idx = tid_y * floats_per_thread * 4;

    #pragma unroll
    for (int i = 0; i < floats_per_thread; i++) {
        const int d = start_idx + i * 4;
        if (d < D_HEAD) {
            float4* s_O_float4 = reinterpret_cast<float4*>(&s_O[tid_x * (D_HEAD + 4) + d]);
            *s_O_float4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
}

// Output normalization and writing function, used only in fwd, but can be shared among different implementations of mta
template<int BLOCK_SIZE_X, int BLOCK_SIZE_Y, int D_HEAD>
__device__ void normalize_and_write_output(
    float* O,
    float* M,
    float* L,
    float* s_O,
    float* s_row_max,
    float* s_row_sum,
    int qkv_offset,
    int ml_offset,
    int q_idx,
    int seq_len,
    int pad_q,
    int tid_x,
    int tid_y
) {
    if (q_idx >= 0 && q_idx < seq_len && tid_x >= pad_q) {
        float row_sum = s_row_sum[tid_x];
        const float inv_row_sum = (row_sum > 1e-10f) ? (1.0f / row_sum) : 0.0f;

        float4* O_vec = (float4*)&O[qkv_offset + q_idx * D_HEAD];

        const int vec_elements_per_thread = (D_HEAD / 4) / BLOCK_SIZE_Y;
        const int vec_start = tid_y * vec_elements_per_thread;

        #pragma unroll
        for (int d = 0; d < vec_elements_per_thread; d++) {
            float4 s_O_val = ((float4*)&s_O[tid_x * (D_HEAD + 4)])[vec_start + d];
            float4 norm_val = make_float4(s_O_val.x * inv_row_sum, s_O_val.y * inv_row_sum, s_O_val.z * inv_row_sum, s_O_val.w * inv_row_sum);
            O_vec[vec_start + d] = norm_val;
        }

        if ((D_HEAD / 4) % BLOCK_SIZE_Y != 0 && tid_y == 0) {
            int remaining = (D_HEAD / 4) % BLOCK_SIZE_Y;
            for (int r = 0; r < remaining; r++) {
                float4 s_O_val = ((float4*)&s_O[tid_x * (D_HEAD + 4)])[D_HEAD / 4 - remaining + r];
                float4 norm_val = make_float4(s_O_val.x * inv_row_sum, s_O_val.y * inv_row_sum, s_O_val.z * inv_row_sum, s_O_val.w * inv_row_sum);
                O_vec[D_HEAD / 4 - remaining + r] = norm_val;
            }
        }
    }

    if (tid_y == 0 && q_idx >= 0 && q_idx < seq_len && tid_x >= pad_q) {
        M[ml_offset + q_idx] = s_row_max[tid_x];
        L[ml_offset + q_idx] = s_row_sum[tid_x];
    }
}
