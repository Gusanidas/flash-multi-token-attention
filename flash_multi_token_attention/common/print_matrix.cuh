#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cmath> // Added for fabsf

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cmath>

// Debugging function: prints a rectangular matrix in quadrants with statistics for inspection.
template<typename T>
__device__ inline void print_rectangular_matrix_in_quadrants(
    const T* matrix,
    const char* matrix_name,
    int rows,
    int cols,
    int leading_dim,
    float scale_factor = 1.0f,
    int tid_linear = 0,
    int seq_len = 0,
    int qk_block_idx = 0
) {
    if (tid_linear == 0 && seq_len <= 33 && qk_block_idx == 0) {
        int print_cols = (cols > 32) ? 32 : cols;
        
        printf("%s (ROWS=%d, COLS=%d", matrix_name, rows, print_cols);
        if (cols > 32) {
            printf(", showing first 32 of %d cols", cols);
        }
        printf("):\n");

        // --- Calculate statistics ---
        float max_abs = 0.0f;
        double sum_abs = 0.0;
        int valid_elements = 0;

        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < print_cols; ++col) {
                T val = matrix[row * leading_dim + col];
                float float_val;
                if constexpr (std::is_same_v<T, float>) {
                    float_val = val;
                } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                    float_val = __bfloat162float(val);
                }
                
                float current_abs = fabsf(scale_factor * float_val);
                sum_abs += current_abs;
                if (current_abs > max_abs) {
                    max_abs = current_abs;
                }
                valid_elements++;
            }
        }
        float mean_abs = (valid_elements > 0) ? static_cast<float>(sum_abs / valid_elements) : 0.0f;
        
        // Calculate quadrant boundaries
        int mid_row = rows / 2;
        int mid_col = print_cols / 2;
        
        // Helper lambda to print a quadrant
        auto print_quadrant = [&](const char* quad_name, int start_row, int end_row, int start_col, int end_col) {
            printf("%s quadrant (rows %d-%d, cols %d-%d):\n", quad_name, start_row, end_row-1, start_col, end_col-1);
            
            // Print column headers
            printf("      ");
            for (int col = start_col; col < end_col; ++col) {
                printf("  %3d  ", col);
            }
            printf("\n");
            
            // Print rows
            for (int row = start_row; row < end_row; ++row) {
                printf("Row %2d:", row);
                for (int col = start_col; col < end_col; ++col) {
                    T val = matrix[row * leading_dim + col];
                    if constexpr (std::is_same_v<T, float>) {
                        printf(" %6.2f", scale_factor * val);
                    } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                        printf(" %6.2f", scale_factor * __bfloat162float(val));
                    }
                }
                printf("\n");
            }
            printf("\n");
        };
        
        // Print quadrants (handle cases where dimensions might be odd)
        if (mid_row > 0 && mid_col > 0) {
            // Top-left quadrant
            print_quadrant("Top-left", 0, mid_row, 0, mid_col);
            
            // Top-right quadrant (if there are columns in the right half)
            if (mid_col < print_cols) {
                print_quadrant("Top-right", 0, mid_row, mid_col, print_cols);
            }
            
            // Bottom-left quadrant (if there are rows in the bottom half)
            if (mid_row < rows) {
                print_quadrant("Bottom-left", mid_row, rows, 0, mid_col);
            }
            
            // Bottom-right quadrant (if there are both bottom rows and right columns)
            if (mid_row < rows && mid_col < print_cols) {
                print_quadrant("Bottom-right", mid_row, rows, mid_col, print_cols);
            }
        } else {
            // Handle edge cases where matrix is too small for quadrants
            printf("Matrix too small for quadrants, printing full matrix:\n");
            print_quadrant("Full", 0, rows, 0, print_cols);
        }

        // Print statistics
        printf("----------------------------------------\n");
        printf("Statistics for %s:\n", matrix_name);
        printf("  - Max absolute value:  %f\n", max_abs);
        printf("  - Mean absolute value: %f\n", mean_abs);
        printf("  - Elements analyzed:   %d\n\n", valid_elements);
    }
}

// Convenience wrapper that maintains the original interface for square matrices
template<typename T>
__device__ inline void print_matrix_in_quadrants(
    const T* matrix,
    const char* matrix_name,
    int block_size_x,
    int leading_dim,
    float scale_factor = 1.0f,
    int d_head = -1,
    int tid_linear = 0,
    int seq_len = 0,
    int qk_block_idx = 0
) {
    int effective_cols = (d_head > 0) ? d_head : block_size_x;
    print_rectangular_matrix_in_quadrants(
        matrix, matrix_name, block_size_x, effective_cols, leading_dim,
        scale_factor, tid_linear, seq_len, qk_block_idx
    );
}