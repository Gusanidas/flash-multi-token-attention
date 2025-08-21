#include <torch/extension.h>
#include <cuda_runtime.h>
#include <float.h>
#include <cstdio>
#include <cuda_bf16.h>
#include <mma.h>
#include <cooperative_groups.h>
#include <algorithm> 

#include "../../common/wmma_utils.cuh"
#include "../../common/conv_utils.cuh"
#include "../../common/kernels/common.h"
#include "../../common/print_matrix.cuh"

using namespace nvcuda;
using namespace cooperative_groups;

template<int BLOCK_SIZE_X, int BLOCK_SIZE_Y, int KX, int KY, int D_HEAD>
__global__ void dk_dv_kernel(
    const __nv_bfloat16* __restrict__ Q,          // [batch, heads, seq_len, d_head]
    const __nv_bfloat16* __restrict__ K,          // [batch, heads, seq_len, d_head]
    const __nv_bfloat16* __restrict__ V,          // [batch, heads, seq_len, d_head]
    const __nv_bfloat16* __restrict__ dO,         // [batch, heads, seq_len, d_head]
    const __nv_bfloat16* __restrict__ D,          // [batch, heads, seq_len]
    const __nv_bfloat16* __restrict__ L,          // [batch, heads, seq_len]
    const __nv_bfloat16* __restrict__ conv_kernel, // [heads, kx, ky]
    float* __restrict__ dK,                       // [batch, heads, seq_len, d_head] - output
    float* __restrict__ dV,                       // [batch, heads, seq_len, d_head] - output
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const float scale,
    const int pad_q,
    const int pad_k,
    const bool causal
) {
    // Thread and block indices
    const int tid_x = threadIdx.x;
    const int tid_y = threadIdx.y;
    const int tid_linear = tid_y * BLOCK_SIZE_X + tid_x;
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int total_threads = BLOCK_SIZE_X * BLOCK_SIZE_Y;
    const int num_warps = total_threads / 32;

    // --- Indexing Strategy ---
    const int reduced_block_size_q = BLOCK_SIZE_X - 2 * pad_q;
    const int reduced_block_size_k = BLOCK_SIZE_X - 4 * pad_k;

    const int k_block_idx = blockIdx.x;
    const int k_start = k_block_idx * reduced_block_size_k - 2 * pad_k;
    const int k_idx = k_start + tid_x;

    // Offsets
    const int qkv_offset = (batch_idx * num_heads + head_idx) * seq_len * D_HEAD;
    const int conv_kernel_offset = head_idx * KX * KY;
    const int ml_offset = (batch_idx * num_heads + head_idx) * seq_len;

    // Padded dimension to avoid bank conflicts
    constexpr int D_HEAD_PADDED = D_HEAD + 4;
    constexpr int QK_LDA = BLOCK_SIZE_X + 2;

    // Shared memory allocation
    extern __shared__ float shared_mem[];
    float* shared_mem_float = shared_mem;

    // Calculate sizes in floats
    constexpr int SIZE_ACCUM_F = BLOCK_SIZE_X * D_HEAD_PADDED;
    constexpr int SIZE_QK_F = BLOCK_SIZE_X * QK_LDA;
    constexpr int SIZE_DP_F = BLOCK_SIZE_X * BLOCK_SIZE_X;
    // R1 size: Max(Accumulator, QK)
    constexpr int SIZE_R1 = (SIZE_ACCUM_F > SIZE_QK_F) ? SIZE_ACCUM_F : SIZE_QK_F;

    // R1: Shared between Accumulator (dK or dV sequentially) and QK
    float* s_R1 = shared_mem_float;
    float* s_Accum = s_R1;
    float* s_QK = s_R1;

    // R2: Shared between dP and dS
    float* s_R2 = s_R1 + SIZE_R1;
    float* s_dP = s_R2;
    float* s_dS = s_R2;

    float* s_D = s_R2 + SIZE_DP_F;
    float* s_L = s_D + BLOCK_SIZE_X;

    // BF16 Buffers (Start after floats)
    __nv_bfloat16* shared_mem_bf16 = (__nv_bfloat16*)(s_L + BLOCK_SIZE_X);

    constexpr int SIZE_BXD_BF16 = BLOCK_SIZE_X * D_HEAD_PADDED;
    constexpr int SIZE_BXB_BF16 = BLOCK_SIZE_X * BLOCK_SIZE_X;

    // Buffer A (Shared by Q and dO)
    __nv_bfloat16* s_A = shared_mem_bf16;
    // Buffer B (Shared by K and V)
    __nv_bfloat16* s_B = s_A + SIZE_BXD_BF16;

    // P/dQK (Shared)
    __nv_bfloat16* s_P = s_B + SIZE_BXD_BF16;
    __nv_bfloat16* s_dQK = s_P;

    // Transpose Buffer (Required because P/dQK might be needed while transposing the other)
    __nv_bfloat16* s_Transpose = s_P + SIZE_BXB_BF16;

    // Conv Kernel
    __nv_bfloat16* s_conv_kernel = s_Transpose + SIZE_BXB_BF16;


    const int kernel_size = KX * KY;
    for (int i = tid_linear; i < kernel_size; i += total_threads) {
        s_conv_kernel[i] = conv_kernel[conv_kernel_offset + i];
    }

    // --- Setup for WMMA and Convolution Groups ---
    const int warp_id = tid_linear / 32;
    constexpr int GROUP_SIZE = 8;
    constexpr int elements_per_thread = BLOCK_SIZE_X / GROUP_SIZE;

    // Setup cooperative groups
    thread_block block = this_thread_block();
    thread_block_tile<32> warp_tile = tiled_partition<32>(block);
    thread_block_tile<GROUP_SIZE> row_group = tiled_partition<GROUP_SIZE>(warp_tile);

    const int row_group_id = tid_linear / GROUP_SIZE;
    const int thread_in_group = tid_linear % GROUP_SIZE;
    const int total_groups = total_threads / GROUP_SIZE;
    const int rows_per_group = (BLOCK_SIZE_X + total_groups - 1) / total_groups;

    // Helper lambda for GMEM R/W operations (dK or dV)
    // This handles reading/writing the accumulators (R1) from/to Global Memory. TODO: Either use the common function or merge it with this one
    auto accumulator_gm_rw = [&](float* gm_ptr, float* smem_ptr, bool is_load, int iter_idx) {
        bool valid_k = (k_idx >= 0 && k_idx < seq_len && tid_x >= 2*pad_k && tid_x < BLOCK_SIZE_X - 2*pad_k);

        const int vec_elements_per_thread = (D_HEAD / 4) / BLOCK_SIZE_Y;
        const int vec_start = tid_y * vec_elements_per_thread;

        if (is_load) {
            if (iter_idx == 0) {
                initialize_shared_memory<BLOCK_SIZE_X, BLOCK_SIZE_Y, D_HEAD>(
                    nullptr, nullptr, nullptr, smem_ptr, tid_x, tid_y, tid_linear, 0.0f, 0.0f, 0.0f
                );
                return;
            }

            if (!valid_k) {
                 float4* smem_vec_ptr = (float4*)&smem_ptr[tid_x * D_HEAD_PADDED];
                 #pragma unroll
                 for (int d = 0; d < vec_elements_per_thread; d++) {
                    smem_vec_ptr[vec_start + d] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                 }
            }
        }

        if (valid_k) {
            float4* gm_vec_ptr = (float4*)&gm_ptr[qkv_offset + k_idx * D_HEAD];
            float4* smem_vec_ptr = (float4*)&smem_ptr[tid_x * D_HEAD_PADDED];

            #pragma unroll
            for (int d = 0; d < vec_elements_per_thread; d++) {
                int idx = vec_start + d;
                if (is_load) {
                    smem_vec_ptr[idx] = gm_vec_ptr[idx];
                } else {
                    // Store to GMEM (unscaled)
                    gm_vec_ptr[idx] = smem_vec_ptr[idx];
                }
            }
        }
    };


    const int num_q_blocks = (seq_len + reduced_block_size_q - 1) / reduced_block_size_q;
    for (int q_block_idx = 0; q_block_idx < num_q_blocks; q_block_idx++) {

        const int q_start = q_block_idx * reduced_block_size_q - 2 * pad_q;
        const int q_idx = q_start + tid_x;

        // 1. Load Q (s_A), K (s_B), L, D.
        // We reload K and Q here as they share buffers with dO and V.
        load_matrix<BLOCK_SIZE_Y, D_HEAD>(Q, s_A, q_idx, seq_len, qkv_offset, tid_x, tid_y);
        load_matrix<BLOCK_SIZE_Y, D_HEAD>(K, s_B, k_idx, seq_len, qkv_offset, tid_x, tid_y);

        if (tid_y == 0) {
            if (q_idx >= 0 && q_idx < seq_len) {
                s_L[tid_x] = __bfloat162float(L[ml_offset + q_idx]);
                s_D[tid_x] = __bfloat162float(D[ml_offset + q_idx]);
            } else {
                s_L[tid_x] = 2000.0f;
                s_D[tid_x] = 0.0f;
            }
        }
        __syncthreads();

        // 2. QK (R1). s_A @ s_B.T -> s_QK.
        wmma_multiply_bf16_to_float<false, false>(
            s_A, s_B, s_QK,
            D_HEAD_PADDED, D_HEAD_PADDED, QK_LDA,
            BLOCK_SIZE_X, BLOCK_SIZE_X, D_HEAD,
            scale, warp_id, num_warps
        );
        __syncthreads();

        // 3. P (s_P). Compute P = exp(conv(QK^T) - L)
        for (int row_offset = 0; row_offset < rows_per_group; row_offset++) {
            const int q_row = row_group_id * rows_per_group + row_offset;
            if (q_row < BLOCK_SIZE_X) {
                const int start_k = thread_in_group * elements_per_thread;
                float rP[elements_per_thread];

                compute_2d_convolution<KX, KY, BLOCK_SIZE_X, elements_per_thread, GROUP_SIZE>(
                    s_QK, s_conv_kernel, QK_LDA,
                    q_row, start_k, row_group, thread_in_group,
                    pad_q, 0, pad_k,
                    q_start, k_start, seq_len, causal, -FLT_MAX, rP
                );

                // P = exp(conv(QK^T) - L)
                float L_val = s_L[q_row];
                #pragma unroll
                for (int i = 0; i < elements_per_thread; i++) {
                    float exp_value = __expf(rP[i] - L_val);
                    int idx = start_k + i + pad_k;

                    if (idx >= 0 && idx < BLOCK_SIZE_X) {
                        float safe_value = isnan(exp_value) ? 0.0f : exp_value;
                        s_P[q_row * BLOCK_SIZE_X + idx] = __float2bfloat16(safe_value);
                    } else {
                        int wrap_idx = (idx + BLOCK_SIZE_X) % BLOCK_SIZE_X;
                        if (wrap_idx < pad_k) {
                                s_P[q_row * BLOCK_SIZE_X + wrap_idx] = __float2bfloat16(0.0f);
                        }
                    }
                }
            }
        }
        // R1 (s_QK) is now free.
        __syncthreads(); // TODO: erase?

        // 4. Load dO (s_A, overwrites Q). Load V (s_B, overwrites K).
        load_matrix<BLOCK_SIZE_Y, D_HEAD>(dO, s_A, q_idx, seq_len, qkv_offset, tid_x, tid_y);
        load_matrix<BLOCK_SIZE_Y, D_HEAD>(V, s_B, k_idx, seq_len, qkv_offset, tid_x, tid_y);
        __syncthreads();

        // 5. dP (R2). s_A @ s_B.T -> s_dP.
        wmma_multiply_bf16_to_float<false, false>(
            s_A, s_B, s_dP,
            D_HEAD_PADDED, D_HEAD_PADDED, BLOCK_SIZE_X,
            BLOCK_SIZE_X, BLOCK_SIZE_X, D_HEAD,
            1.0f, warp_id, num_warps
        );
        __syncthreads();
        

        // 6 Load dV_accum (R1) from GMEM.
        accumulator_gm_rw(dV, s_Accum, true, q_block_idx);
        __syncthreads(); // TODO: erase?

        // 6 Transpose P (s_Transpose).
        for (int i = tid_linear; i < BLOCK_SIZE_X * BLOCK_SIZE_X; i += total_threads) {
            int row = i / BLOCK_SIZE_X;
            int col = i % BLOCK_SIZE_X;
            bool valid = (row >= pad_q) && (row < BLOCK_SIZE_X-pad_q) && (col >= 2*pad_k) && (col < BLOCK_SIZE_X - 2*pad_k);
            if (valid) {
                s_Transpose[col * BLOCK_SIZE_X + row] = s_P[row * BLOCK_SIZE_X + col];
            } else {
                s_Transpose[col * BLOCK_SIZE_X + row] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();


        // 6c. R1 += P.T @ dO (s_A). (load_c=true)
        wmma_multiply_bf16_to_float<true, true>(
            s_Transpose, s_A, s_Accum,
            BLOCK_SIZE_X, D_HEAD_PADDED, D_HEAD_PADDED,
            BLOCK_SIZE_X, D_HEAD, BLOCK_SIZE_X,
            1.0f, warp_id, num_warps
        );
        __syncthreads();

        // 6d. Store dV_accum (R1) to GMEM.
        accumulator_gm_rw(dV, s_Accum, false, q_block_idx);
        

        // 7. dS (R2).
        const int elements_per_thread_ds = (BLOCK_SIZE_X * BLOCK_SIZE_X) / total_threads;
        const int ds_start_idx = tid_linear * elements_per_thread_ds;

        #pragma unroll
        for (int idx = 0; idx < elements_per_thread_ds; idx++) {
            int linear_idx = ds_start_idx + idx;
            if (linear_idx < BLOCK_SIZE_X * BLOCK_SIZE_X) {
                int row = linear_idx / BLOCK_SIZE_X;
                // We still need s_P here.
                float P_val = __bfloat162float(s_P[linear_idx]);
                float dP_val = s_dP[linear_idx];
                float D_val = s_D[row];
                s_dS[linear_idx] = P_val * (dP_val - D_val);
            }
        }
        __syncthreads();


        // 8. dQK (reuses s_P).
        for (int row_offset = 0; row_offset < rows_per_group; row_offset++) {
            const int q_row = row_group_id * rows_per_group + row_offset;

            if (q_row < BLOCK_SIZE_X) {
                const int start_k = thread_in_group * elements_per_thread;
                float r_dQK[elements_per_thread];

                // Transposed convolution (FLIP_KERNEL=true)
                compute_2d_convolution<KX, KY, BLOCK_SIZE_X, elements_per_thread, GROUP_SIZE, true>(
                    s_dS, s_conv_kernel, BLOCK_SIZE_X,
                    q_row, start_k, row_group, thread_in_group,
                    0, pad_q, 0,
                    0, 0, BLOCK_SIZE_X,
                    false, 0.0f, r_dQK
                );

                // Write results to s_dQK (S5) with boundary checks
                #pragma unroll
                for (int i = 0; i < elements_per_thread; i++) {
                    int global_pos = start_k + i + pad_k;
                    if (global_pos < BLOCK_SIZE_X && q_row < BLOCK_SIZE_X) {
                        int q_pos = q_start + q_row;
                        int k_pos = global_pos + k_start;

                        bool valid =
                            (q_pos >= 0) && (q_pos < seq_len) &&
                            (k_pos >= 0) && (k_pos < seq_len) &&
                            (q_row >= pad_q) && (q_row < BLOCK_SIZE_X-pad_q) &&
                            (global_pos < BLOCK_SIZE_X - 2*pad_k) &&
                            (global_pos >= 2*pad_k);

                        if (causal) {
                            valid = valid && (k_pos <= q_pos);
                        }

                        if (valid) {
                            s_dQK[q_row * BLOCK_SIZE_X + global_pos] = __float2bfloat16(r_dQK[i]);
                        } else {
                            s_dQK[q_row * BLOCK_SIZE_X + global_pos] = __float2bfloat16(0.0f);
                        }
                    }
                }
            }
        }
        // R2 (s_dS) is free.
        __syncthreads(); // TODO: erase?
        

        // 9. dK accumulation.
        // 9a. Load Q (s_A, overwrites dO).
        load_matrix<BLOCK_SIZE_Y, D_HEAD>(Q, s_A, q_idx, seq_len, qkv_offset, tid_x, tid_y);

        // 9b. Load dK_accum (R1) from GMEM.
        accumulator_gm_rw(dK, s_Accum, true, q_block_idx);
        __syncthreads();

        // 9c. Transpose dQK (s_Transpose).
        for (int i = tid_linear; i < BLOCK_SIZE_X * BLOCK_SIZE_X; i += total_threads) {
            int row = i / BLOCK_SIZE_X;
            int col = i % BLOCK_SIZE_X;
            s_Transpose[col * BLOCK_SIZE_X + row] = s_dQK[row * BLOCK_SIZE_X + col];
        }
        __syncthreads();

        // 9d. R1 += dQK.T @ Q (s_A). (load_c=true)
        wmma_multiply_bf16_to_float<true, true>(
            s_Transpose, s_A, s_Accum,
            BLOCK_SIZE_X, D_HEAD_PADDED, D_HEAD_PADDED,
            BLOCK_SIZE_X, D_HEAD, BLOCK_SIZE_X,
            1.0f, warp_id, num_warps
        );
        __syncthreads();

        // 9e. Store dK_accum (R1) to GMEM.
        accumulator_gm_rw(dK, s_Accum, false, q_block_idx);


        // Sync before starting the next iteration
        __syncthreads();
    } 


    // --- Finalization: Apply scale to dK ---
    // dK was accumulated in GMEM without the scale factor. We apply it now.
    // dV is already finalized in GMEM (unscaled).

    bool valid_k = (k_idx >= 0 && k_idx < seq_len && tid_x >= 2*pad_k && tid_x < BLOCK_SIZE_X - 2*pad_k);

    if (valid_k) {
        float4* dK_vec = (float4*)&dK[qkv_offset + k_idx * D_HEAD];

        const int vec_elements_per_thread = (D_HEAD / 4) / BLOCK_SIZE_Y;
        const int vec_start = tid_y * vec_elements_per_thread;

        #pragma unroll
        for (int d = 0; d < vec_elements_per_thread; d++) {
            int vec_idx = vec_start + d;

            // Read dK (accumulated without scale), apply scale factor, write back.
            float4 dk_val = dK_vec[vec_idx];
            dk_val.x *= scale;
            dk_val.y *= scale;
            dk_val.z *= scale;
            dk_val.w *= scale;
            dK_vec[vec_idx] = dk_val;
        }
    }
}

template<int KX, int KY, int D_HEAD>
void launch_dkdv_backward_kernel(
    torch::Tensor Q,
    torch::Tensor K,
    torch::Tensor V,
    torch::Tensor dO,
    torch::Tensor D,
    torch::Tensor L,
    torch::Tensor conv_kernel,
    torch::Tensor dK,
    torch::Tensor dV,
    float scale,
    int pad_q,
    int pad_k,
    bool causal
) {
    const int batch_size = Q.size(0);
    const int num_heads = Q.size(1);
    const int seq_len = Q.size(2);

    
    // Use the same block size as dq_kernel for consistency
    const int BLOCK_SIZE_X = 48; // TODO: make it flexible
    const int BLOCK_SIZE_Y = 4;
    
    // Calculate grid dimensions
    const int reduced_k_block_size = BLOCK_SIZE_X - 4*pad_k;
    const int num_blocks_k = (seq_len + reduced_k_block_size - 1) / reduced_k_block_size;
    
    dim3 grid(num_blocks_k, num_heads, batch_size);
    dim3 block(BLOCK_SIZE_X, BLOCK_SIZE_Y);

    // --- Calculate Optimized Shared Memory Size (Interleaved Layout) ---
    const int D_HEAD_PADDED = D_HEAD + 4;
    const int QK_LDA = BLOCK_SIZE_X + 2;

    // Float sizes (in bytes)
    const int SIZE_ACCUM_BYTES = BLOCK_SIZE_X * D_HEAD_PADDED * sizeof(float);
    const int SIZE_QK_BYTES = BLOCK_SIZE_X * QK_LDA * sizeof(float);
    const int SIZE_DP_BYTES = BLOCK_SIZE_X * BLOCK_SIZE_X * sizeof(float);
    const int SIZE_DL_BYTES = 2 * BLOCK_SIZE_X * sizeof(float); // D and L

    // R1: Max(Accumulator, QK)
    const int R1_BYTES = std::max(SIZE_ACCUM_BYTES, SIZE_QK_BYTES);
    // R2: dP/dS
    const int R2_BYTES = SIZE_DP_BYTES;

    const int total_float_size = R1_BYTES + R2_BYTES + SIZE_DL_BYTES;

    // BF16 sizes (in bytes)
    const int SIZE_BXD_BF16_BYTES = BLOCK_SIZE_X * D_HEAD_PADDED * sizeof(__nv_bfloat16);
    const int SIZE_BXB_BF16_BYTES = BLOCK_SIZE_X * BLOCK_SIZE_X * sizeof(__nv_bfloat16);
    const int SIZE_KERNEL_BYTES = KX * KY * sizeof(__nv_bfloat16);

    // Buffer A (Q/dO) + Buffer B (K/V)
    const int BUFF_AB_BYTES = 2 * SIZE_BXD_BF16_BYTES;
    // P/dQK
    const int P_BYTES = SIZE_BXB_BF16_BYTES;
    // Transpose
    const int TRANSPOSE_BYTES = SIZE_BXB_BF16_BYTES;

    const int total_bf16_size = BUFF_AB_BYTES + P_BYTES + TRANSPOSE_BYTES + SIZE_KERNEL_BYTES;

    // Total size calculation. Alignment is handled by placing BF16 after Floats.
    const int shared_mem_size = total_float_size + total_bf16_size;

    //printf("dkdv optimized shared_mem_size: %d\n", shared_mem_size);
    //printf("Total Float size: %d\n", total_float_size);
    //printf("  R1 (Accum/QK): %d\n", R1_BYTES);
    //printf("  R2 (dP/dS): %d\n", R2_BYTES);
    //printf("Total BF16 size: %d\n", total_bf16_size);
    //printf("  Buffers A+B (Q/dO, K/V): %d\n", BUFF_AB_BYTES);
    //printf("  P/dQK + Transpose: %d\n", P_BYTES + TRANSPOSE_BYTES);

    // Configure shared memory if necessary (e.g., for > 48KB on modern GPUs)
    int device;
    cudaGetDevice(&device);
    int max_shared_mem;
    // Use Optin attribute to check the maximum possible dynamic shared memory.
    cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    if (shared_mem_size > 49152) {
        if (shared_mem_size > max_shared_mem) {
             TORCH_CHECK(false, "dk_dv_kernel requires ", shared_mem_size, " bytes of shared memory, but max available is ", max_shared_mem, " bytes. Reduce BLOCK_SIZE_X or D_HEAD.");
        }
        // Opt-in to required dynamic shared memory size
        cudaError_t res = cudaFuncSetAttribute(dk_dv_kernel<BLOCK_SIZE_X, BLOCK_SIZE_Y, KX, KY, D_HEAD>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
         if (res != cudaSuccess) {
            TORCH_CHECK(false, "Failed to set shared memory attribute: ", cudaGetErrorString(res));
        }
    }

    dk_dv_kernel<BLOCK_SIZE_X, BLOCK_SIZE_Y, KX, KY, D_HEAD><<<grid, block, shared_mem_size>>>(
        reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(V.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(dO.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(D.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(L.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(conv_kernel.data_ptr()),
        reinterpret_cast<float*>(dK.data_ptr()),
        reinterpret_cast<float*>(dV.data_ptr()),
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