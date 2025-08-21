#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

/**
 * Common function to multiply two matrices in shared memory using WMMA.
 * Multiplies bfloat16 matrices A and B, writes result to shared memory in float.
 * 
 * @param s_A: Input matrix A in shared memory (bfloat16, row-major)
 * @param s_B: Input matrix B in shared memory (bfloat16, row-major for A, col-major for B in Q*K^T)
 * @param s_C: Output matrix C in shared memory (float, row-major)
 * @param lda: Leading dimension of matrix A
 * @param ldb: Leading dimension of matrix B  
 * @param ldc: Leading dimension of matrix C
 * @param M: Number of rows in A and C
 * @param N: Number of columns in B and C
 * @param K: Number of columns in A and rows in B
 * @param scale: Scaling factor to apply to result
 * @param b_row_major: Whether matrix B is row-major (true) or column-major (false)
 * @param warp_id: Warp ID for tile assignment
 * @param num_warps: Total number of warps participating
 * @param load_c: Whether to load existing values from C (true) or initialize to zero (false)
 */
template<bool load_c = false, bool b_row_major = false>
__device__ void wmma_multiply_bf16_to_float(
    const __nv_bfloat16* s_A,
    const __nv_bfloat16* s_B, 
    float* s_C,
    int lda,
    int ldb,
    int ldc,
    int M,
    int N, 
    int K,
    float scale = 1.0f,
    int warp_id = 0,
    int num_warps = 1
) {
    // Process matrices in 16x16 tiles (hardcoded WMMA dimensions)
    const int WMMA_M = 16;
    const int WMMA_N = 16; 
    const int WMMA_K = 16;
    
    int total_tiles = (M / WMMA_M) * (N / WMMA_N);
    
    for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += num_warps) {
        int tile_a = tile_idx % (M / WMMA_M);
        int tile_b = tile_idx / (M / WMMA_M);
        
        // Initialize WMMA fragments - all threads in warp must participate
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
            
        // Initialize or load accumulator based on template parameter
        if constexpr (load_c) {
            // Load existing values from shared memory C
            wmma::load_matrix_sync(acc_frag, &s_C[(tile_a * WMMA_M) * ldc + (tile_b * WMMA_N)], ldc, wmma::mem_row_major);
        } else {
            // Initialize accumulator to zero
            wmma::fill_fragment(acc_frag, 0.0f);
        }
            
        // Process across K dimension in chunks of WMMA_K
        for (int k_tile = 0; k_tile < K / WMMA_K; k_tile++) {
            int k_start = k_tile * WMMA_K;
            int a_start_tile = tile_a * WMMA_M;
            int b_start_tile = tile_b * WMMA_N;
            
            // Load A fragment (always row-major)
            wmma::load_matrix_sync(a_frag, &s_A[a_start_tile * lda + k_start], lda);
            
            // Load B fragment with specified layout using template parameter
            if constexpr (b_row_major) {
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag, &s_B[k_start * ldb + b_start_tile], ldb);
                // Perform matrix multiplication  
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            } else {
                wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> b_frag;
                wmma::load_matrix_sync(b_frag, &s_B[b_start_tile * ldb + k_start], ldb);
                // Perform matrix multiplication
                wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
            }
        }
            
        // Apply scaling to accumulator
        if (scale != 1.0f) {
            #pragma unroll
            for (int i = 0; i < acc_frag.num_elements; i++) {
                acc_frag.x[i] *= scale;
            }
        }
            
        // Store result to output matrix
        wmma::store_matrix_sync(
            &s_C[(tile_a * WMMA_M) * ldc + (tile_b * WMMA_N)], 
            acc_frag, 
            ldc,
            wmma::mem_row_major
        );
    }
}