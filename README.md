Implementations of multi token attention in CUDA and Triton.

- (Paper)[https://arxiv.org/abs/2504.00927]
- (Original implementation)[https://github.com/facebookresearch/RAM/tree/main/projects/mta]


## Conv Attention ##

In this variant, there is only a convolution before the softmax. 

$$
A = \text{Softmax} \ \text{Conv2d}_{\theta}(\hat{A})
$$

$$
a_{ij} = \text{Softmax} \left( \frac{c_{q-1}}{\sqrt{d}} \sum_{i'=0}^{\lceil c_k/2 \rceil-1} \sum_{j'=-\lfloor c_k/2 \rfloor}^{\lfloor c_k/2 \rfloor} 1_{i \ge j-j'} \theta_{i',j'} q_{i-i'}^{\top} k_{j-j'} \right)
$$

And the output:

```math
\text{Out}_{jh} = \left( \sum a_{ji} \dot V_{ih} \right)
```


The implementation fuses all these operations in a single CUDA kernel, using the main ideas of tiling and online softmax from Flash Attention.

Because of the convolution, the tiles overlap each other, and there are some values of the Q*K^T matrix that are recomputed.


This kernel achieves some speedup compared with the pytorch implementation:
![speed-up1](benchmark_seq_len_speedup.png)

Because of the unavoidable extra computations involved in the convolution, it is slower than standard flash attention in the training workload:

![fwd_conv](benchmark_seq_len_backward_times.png)
![bwd_conv](benchmark_seq_len_forward_times.png)

However, during inference, memory bandwith is the bottleneck and increasing arithmetic intensity doesnt affect the performance as much:

![fwd_inference](fwd_inference_speedup.png)

### Algorithm 1: Fused ConvAttention Forward (Overlapping Tiles Implementation)
**Require**: Matrices $Q, K, V \in \mathbb{R}^{N \times d}$, Convolution Kernel $W$.
**Parameters**: SRAM tile size $B$. Padding $P_q, P_k$.

1. Initialize $O=(0)_{N \times d}$, $L=(0)_N$, $M=(-\infty)_N$ in HBM.
2. Define effective block sizes (strides): $B_r = B - P_q$, $B_c = B - 2P_k$.
3. Calculate number of blocks: $T_r = \lceil N/B_r \rceil, T_c = \lceil N/B_c \rceil$.
4. **Parallelize for** $i=1$ to $T_r$ **do**:
5. &emsp;// Define indices for the loaded Q tile (size $B$)
6. &emsp; $I_s = (i-1) \cdot B_r - P_q$. $I_e = I_s + B$.
7. &emsp;// Define indices for the effective output region (size $B_r$)
8. &emsp; $I_s^{\text{eff}} = (i-1) \cdot B_r. I_e^{\text{eff}} = I_s^{\text{eff}} + B_r$.
9. &emsp;Load $W$ from HBM to SRAM.
10. &emsp;Initialize local statistics in SRAM (Size $B$): $\ell_i = (1)_B$, $m_i = (-\infty)_B$.
11. &emsp;**for** $j=1$ to $T_c$ **do**:
12. &emsp;&emsp;// Define indices for the loaded K/V tile (size $B$)
13. &emsp;&emsp; $J_s = (j-1) \cdot B_c - P_k$. $J_e = J_s + B$
14. &emsp;&emsp;// Load Overlapping Tiles
15. &emsp;&emsp;Load $Q[I_s:I_e], K[J_s:J_e], V[J_s:J_e]$ from HBM to SRAM. (Handle boundary indices).
16. &emsp;&emsp;// Compute Attention Scores on Full Tile
17. &emsp;&emsp;On chip, $S_{ij} = \text{scale} \cdot (Q[I_s:I_e] K[J_s:J_e]^T) \in \mathbb{R}^{B \times B}$.
18. &emsp;&emsp;// Convolution and Local Statistics
19. &emsp;&emsp;On chip, $P_{ij}^{\text{raw}} = \text{Conv2D}(S_{ij}, W) \in \mathbb{R}^{B \times B}$. (Apply causal masking).
20. &emsp;&emsp;On chip, compute local stats:
21. &emsp;&emsp; $\tilde{m}_{ij}$.
22. &emsp;&emsp; $\tilde{P}_{ij}$.
23. &emsp;&emsp; $\tilde{\ell}_{ij}$.
24. &emsp;&emsp;// Online Softmax Update
25. &emsp;&emsp; $m_i^{\text{new}} = \max(m_i, \tilde{m}_{ij})$
26. &emsp;&emsp;Calculate rescaling factors: $\alpha = e^{m_i - m_i^{\text{new}}}$, $\beta = e^{\tilde{m}_{ij} - m_i^{\text{new}}}$.
27. &emsp;&emsp;Update $\ell_i \leftarrow \alpha \cdot \ell_i + \beta \cdot \tilde{\ell}_{ij}$.
28. &emsp;&emsp;// Update Output (Read-Modify-Write HBM)
29. &emsp;&emsp;Load the full tile $O[I_s:I_e]$ from HBM to SRAM.
30. &emsp;&emsp;On chip, $O[I_s:I_e] \leftarrow \alpha \cdot O[I_s:I_e] + (\beta \cdot \tilde{P}_{ij})V[J_s:J_e]$.
31. &emsp;&emsp;Write back only the effective region $O[I_s^{\text{eff}}:I_e^{\text{eff}}]$ to HBM.
&emsp;&emsp;&emsp;&emsp;&emsp;(Corresponds to the slice $[P_q:B]$ of the SRAM buffer).
32. &emsp;&emsp; $m_i \leftarrow m_i^{\text{new}}$
33. &emsp;**end for**
34. &emsp;// Finalization (Load, Normalize, Write effective region)
35. &emsp;Load the accumulated $O[I_s^{\text{eff}}:I_e^{\text{eff}}]$ from HBM.
36. &emsp;Normalize using the corresponding slice of the local statistics (held in SRAM):
&emsp;&emsp; $O[I_s^{\text{eff}}:I_e^{\text{eff}}] \leftarrow O[I_s^{\text{eff}}:I_e^{\text{eff}}] / \ell_i[P_q:B]$.
37. &emsp;Write normalized $O[I_s^{\text{eff}}:I_e^{\text{eff}}]$ to HBM.
38. &emsp;Write $m_i[P_q:B]$ to $M[I_s^{\text{eff}}:I_e^{\text{eff}}]$ and $\ell_i[P_q:B]$ to $L[I_s^{\text{eff}}:I_e^{\text{eff}}]$ in HBM.
39. **end Parallelize**



### Algorithm 2: Fused ConvAttention Backward - dQ and dW (dq_kernel)
**Require**: Matrices $Q, K, V, dO \in \mathbb{R}^{N \times d}$. Statistics $L, D \in \mathbb{R}^{N}$. Convolution Kernel $W$.
**Parameters**: SRAM tile size $B$. Padding $P_q, P_k$. Scale factor $\text{scale}$.

1. Initialize $dQ=(0)_{N \times d}, dW=(0)$ in HBM.
2. Define effective block sizes (strides): $B_r = B - 2P_q, B_c = B - 4P_k$.
3. Calculate number of blocks: $T_r = \lceil N/B_r \rceil, T_c = \lceil N/B_c \rceil$.
4. **Parallelize for** $i=1$ to $T_r$ **do**:
5. &emsp;// Define indices for the loaded Q/dO tile (size $B$)
6. &emsp; $I_s = (i-1) \cdot B_r - 2P_q$. $I_e = I_s + B$.
7. &emsp;// Define indices for the effective output region (size $B_r$)
8. &emsp; $I_s^{\text{eff}} = (i-1) \cdot B_r$. $I_e^{\text{eff}} = I_s^{\text{eff}} + B_r$.
9. &emsp;Load $W$ from HBM to SRAM. Initialize local $dW_i=(0)$ accumulator in SRAM.
10. &emsp;Load $dO[I_s:I_e], L[I_s:I_e], D[I_s:I_e]$ from HBM to SRAM.
11. &emsp;**for** $j=1$ to $T_c$ **do**:
12. &emsp;&emsp;// Define indices for the loaded K/V tile (size $B$)
13. &emsp;&emsp; $J_s = (j-1) \cdot B_c - 2P_k$. $J_e = J_s + B$.
14. &emsp;&emsp;// --- Forward Computation Reprise ---
15. &emsp;&emsp;Load $Q[I_s:I_e], K[J_s:J_e]$ from HBM to SRAM.
16. &emsp;&emsp;On chip, $S_{ij} = \text{scale} \cdot (Q[I_s:I_e] K[J_s:J_e]^T) \in \mathbb{R}^{B \times B}$.
17. &emsp;&emsp; $P_{ij}^{\text{raw}} = \text{Conv2D}(S_{ij}, W)$. (Apply causal masking).
18. &emsp;&emsp; $P_{ij} = \exp(P_{ij}^{\text{raw}} - L[I_s:I_e])$.
19. &emsp;&emsp;// --- Backward Computation ---
20. &emsp;&emsp;// Compute dP
21. &emsp;&emsp;Load $V[J_s:J_e]$ from HBM to SRAM.
22. &emsp;&emsp;On chip, $dP_{ij} = dO[I_s:I_e] V[J_s:J_e]^T$.
23. &emsp;&emsp;// Compute dS (Backprop through Softmax)
24. &emsp;&emsp;On chip, $dS_{ij}^{\text{conv}} = P_{ij} \odot (dP_{ij} - D[I_s:I_e])$.
25. &emsp;&emsp;// Compute dW (Accumulate Kernel Gradient)
26. &emsp;&emsp;On chip, $dW_i \leftarrow dW_i + \text{GradW}(S_{ij}, dS_{ij}^{\text{conv}})$.
27. &emsp;&emsp;// Compute $d(QK^T)$ (Backprop through Convolution)
28. &emsp;&emsp; On chip, $d(QK^T)$ = $\text{TransposedConv2D}(dS^{\text{conv}}, W)$
29. &emsp;&emsp;// Compute dQ (Read-Modify-Write HBM)
30. &emsp;&emsp;Load $K[J_s:J_e]$ from HBM to SRAM (if overwritten by V).
31. &emsp;&emsp;Load $dQ[I_s:I_e]$ from HBM to SRAM accumulator $dQ_{\text{acc}}$. (If $j>1$, else initialize to 0).
32. &emsp;&emsp;On chip, $dQ_{\text{acc}} \leftarrow dQ_{\text{acc}} + d(QK^T)_{ij} K[J_s:J_e]$.
33. &emsp;&emsp;Write back the effective region $dQ[I_s^{\text{eff}}:I_e^{\text{eff}}]$ to HBM (unscaled).
34. &emsp;**end for**
35. &emsp;// Finalization
36. &emsp;// Apply scale factor to dQ (Read-Modify-Write HBM)
37. &emsp;Load $dQ[I_s^{\text{eff}}:I_e^{\text{eff}}]$ from HBM.
38. &emsp; $dQ[I_s^{\text{eff}}:I_e^{\text{eff}}] \leftarrow dQ[I_s^{\text{eff}}:I_e^{\text{eff}}] \cdot \text{scale}$.
39. &emsp;Write $dQ[I_s^{\text{eff}}:I_e^{\text{eff}}]$ back to HBM.
40. &emsp;// Atomically update global dW
41. &emsp;$\text{AtomicAdd}(dW, dW_i)$.
42. **end Parallelize**

### Algorithm 3: Fused ConvAttention Backward - dK and dV (dk_dv_kernel)
**Require**: Matrices $Q, K, V, dO \in \mathbb{R}^{N \times d}$. Statistics $L, D \in \mathbb{R}^{N}$. Convolution Kernel $W$.
**Parameters**: SRAM tile size $B$. Padding $P_q, P_k$. Scale factor $\text{scale}$.

1. Initialize $dK=(0)_{N \times d}, dV=(0)_{N \times d}$ in HBM.
2. Define effective block sizes (strides): $B_r = B - 2P_q, B_c = B - 4P_k$.
3. Calculate number of blocks: $T_r = \lceil N/B_r \rceil, T_c = \lceil N/B_c \rceil$.
4. **Parallelize for** $j=1$ to $T_c$ **do**:
5. &emsp;// Define indices for the loaded K/V tile (size $B$)
6. &emsp; $J_s = (j-1) \cdot B_c - 2P_k$. $J_e = J_s + B$.
7. &emsp;// Define indices for the effective output region (size $B_c$)
8. &emsp; $J_s^{\text{eff}} = (j-1) \cdot B_c$. $J_e^{\text{eff}} = J_s^{\text{eff}} + B_c$.
9. &emsp;Load $W$ from HBM to SRAM.
10. &emsp;**for** $i=1$ to $T_r$ **do**:
11. &emsp;&emsp;// Define indices for the loaded Q/dO tile (size $B$)
12. &emsp;&emsp; $I_s = (i-1) \cdot B_r - 2P_q$. $I_e = I_s + B$.
13. &emsp;&emsp;// --- Forward Computation Reprise ---
14. &emsp;&emsp;Load $Q[I_s:I_e], K[J_s:J_e], L[I_s:I_e], D[I_s:I_e]$ from HBM to SRAM.
15. &emsp;&emsp;On chip, $S_{ij} = \text{scale} \cdot (Q[I_s:I_e] K[J_s:J_e]^T) \in \mathbb{R}^{B \times B}$.
16. &emsp;&emsp; $P_{ij}^{\text{raw}} = \text{Conv2D}(S_{ij}, W)$. (Apply causal masking).
17. &emsp;&emsp; $P_{ij} = \exp(P_{ij}^{\text{raw}} - L[I_s:I_e])$.
18. &emsp;&emsp;// --- Backward Computation (Interleaved dV and dK) ---
19. &emsp;&emsp;// Compute dP
20. &emsp;&emsp;Load $dO[I_s:I_e], V[J_s:J_e]$ from HBM to SRAM (Overwrites Q, K).
21. &emsp;&emsp;On chip, $dP_{ij} = dO[I_s:I_e] V[J_s:J_e]^T$.
22. &emsp;&emsp;// Update dV (Read-Modify-Write HBM)
23. &emsp;&emsp;Load $dV[J_s:J_e]$ from HBM to SRAM accumulator $dV_{\text{acc}}$. (If $i>1$, else initialize to 0).
24. &emsp;&emsp;On chip, transpose $P_{ij} \rightarrow P_{ij}^T$.
25. &emsp;&emsp;On chip, $dV_{\text{acc}} \leftarrow dV_{\text{acc}} + P_{ij}^T dO[I_s:I_e]$.
26. &emsp;&emsp;Write back the effective region $dV[J_s^{\text{eff}}:J_e^{\text{eff}}]$ to HBM.
27. &emsp;&emsp;// Compute dS (Backprop through Softmax)
28. &emsp;&emsp;On chip, $dS_{ij}^{\text{conv}} = P_{ij} \odot (dP_{ij} - D[I_s:I_e])$.
29. &emsp;&emsp;// Compute $d(QK^T)$ (Backprop through Convolution)
30. &emsp;&emsp;On chip, $d({QK^T})$ = $\text{TransposedConv2D}(dS^{\text{conv}}, W)$.
31. &emsp;&emsp;// Update dK (Read-Modify-Write HBM)
32. &emsp;&emsp;Load $Q[I_s:I_e]$ from HBM to SRAM (was overwritten by dO).
33. &emsp;&emsp;Load $dK[J_s:J_e]$ from HBM to SRAM accumulator $dK_{\text{acc}}$. (If $i>1$, else initialize to 0).
34. &emsp;&emsp;On chip, transpose $d(QK^T)_{ij} \rightarrow d(QK^T)_{ij}^T$.
35. &emsp;&emsp;On chip, $dK_{\text{acc}} \leftarrow dK_{\text{acc}} + d(QK^T)_{ij}^T Q[I_s:I_e]$.
36. &emsp;&emsp;Write back the effective region $dK[J_s^{\text{eff}}:J_e^{\text{eff}}]$ to HBM (unscaled).
37. &emsp;**end for**
38. &emsp;// Finalization
39. &emsp;// Apply scale factor to dK (Read-Modify-Write HBM). dV is already finalized.
40. &emsp;Load $dK[J_s^{\text{eff}}:J_e^{\text{eff}}]$ from HBM.
41. &emsp; $dK[J_s^{\text{eff}}:J_e^{\text{eff}}] \leftarrow dK[J_s^{\text{eff}}:J_e^{\text{eff}}] \cdot \text{scale}$.
42. &emsp;Write $dK[J_s^{\text{eff}}:J_e^{\text{eff}}]$ back to HBM.
43. **end Parallelize**

### Algorithm 4: Fused ConvAttention Forward - Inference with K-Splitting
**Require**: Query matrix for the last 16 tokens $Q \in \mathbb{R}^{16 \times d}$, Key matrix $K \in \mathbb{R}^{N_k \times d}$, Value matrix $V \in \mathbb{R}^{N_k \times d}$, Convolution Kernel $W \in \mathbb{R}^{k_x \times k_y}$.
**Parameters**: Number of K-dimension splits $N_{\text{splits}}$, SRAM tile size $B_k$. Scale factor $\text{scale}$.
**Output**: Final attention output vector $O_{\text{final}} \in \mathbb{R}^{1 \times d}$ for the last query token.

1.  // **Part 1: Partial Attention Computation (fwd_inference_kernel)**
2.  // This kernel computes partial attention outputs by splitting the Key/Value matrices along the sequence length dimension ($N_k$).
3.  // Each thread block processes one split for a specific batch and head.
4.  **Parallelize for** each (batch, head, k_split) triplet, where $s=1, \dots, N_{\text{splits}}$ **do**:
5.  &emsp;Initialize partial output $O_s=(0)_{1 \times d}$, partial max $M_s=-\infty$, and partial sum $L_s=0$ in SRAM.
6.  &emsp;Load the entire query tile $Q \in \mathbb{R}^{16 \times d}$ and convolution kernel $W$ from HBM to SRAM.
7.  &emsp;Define the number of K-tiles for this split: $T_k = \lceil N_k / (N_{\text{splits}} \cdot B_k) \rceil$.
8.  &emsp;**for** each K-tile $j=1, \dots, T_k$ **do**:
9.  &emsp;&emsp;// Define indices for the current Key/Value tile of size $B_k$.
10. &emsp;&emsp;Load K tile from HBM to SRAM.
11. &emsp;&emsp;On chip, compute attention scores: $S_j = \text{scale} \cdot (QK^T) \in \mathbb{R}^{16 \times B_k}$.
12. &emsp;&emsp;On chip, apply 2D convolution to the scores: $S_{j, \text{conv}} = \text{Conv2D}(S_{j}, W)$.
13. &emsp;&emsp;Extract the scores corresponding to the last query token: $s_{j, \text{lastQ}} \in \mathbb{R}^{1 \times B_k}$.
14. &emsp;&emsp;// --- Online Softmax Update ---
15. &emsp;&emsp;Find the maximum value in the current tile's scores: $m_j = \max(s_{j, \text{lastQ}})$.
16. &emsp;&emsp;Store the previous max for the split: $M_{\text{old}} = M_s$.
17. &emsp;&emsp;Update the max for the split: $M_s = \max(M_s, m_j)$.
18. &emsp;&emsp;Compute correction factors based on the max update: $\alpha = \exp(M_{\text{old}} - M_s)$ and $\beta = \exp(m_j - M_s)$.
19. &emsp;&emsp;Rescale the previous partial output and sum: $O_s \leftarrow O_s \cdot \alpha$ and $L_s \leftarrow L_s \cdot \alpha$.
20. &emsp;&emsp;Compute softmax probabilities for the current tile: $P_j = \exp(s_{j, \text{lastQ}} - M_s)$.
21. &emsp;&emsp;Update the partial sum: $L_s \leftarrow L_s + \sum(P_j)$.
22. &emsp;&emsp;Load the corresponding V tile from HBM to SRAM.
23. &emsp;&emsp;Update the partial output: $O_s \leftarrow O_s + P_j V$.
24. &emsp;**end for**
25. &emsp;Write the final partial results for this split ($O_s, M_s, L_s$) to HBM.
26. **end Parallelize**
27. // **Part 2: Combine K-Splits (combine_splits_kernel)**
28. // If $N_{\text{splits}} > 1$, this kernel is launched to combine the partial results into a final, correct output.
29. // Each thread block combines the splits for a single (batch, head) pair.
30. **if** $N_{\text{splits}} > 1$ **then**
31. &emsp;**Parallelize for** each (batch, head) pair **do**:
32. &emsp;&emsp;Load all partial results $\{O_s, M_s, L_s\}_{s=1}^{N_{\text{splits}}}$ for the current (batch, head) from HBM to SRAM.
33. &emsp;&emsp;Initialize the combined results with the first split's values: $O_{\text{final}} = O_1, M_{\text{final}} = M_1, L_{\text{final}} = L_1$.
34. &emsp;&emsp;**for** each subsequent split $s=2, \dots, N_{\text{splits}}$ **do**:
35. &emsp;&emsp;&emsp;Store the previous combined max: $M_{\text{old}} = M_{\text{final}}$.
36. &emsp;&emsp;&emsp;Find the new true maximum across the combined and current splits: $M_{\text{final}} = \max(M_{\text{final}}, M_s)$.
37. &emsp;&emsp;&emsp;Compute correction factors: $\alpha = \exp(M_{\text{old}} - M_{\text{final}})$ and $\beta = \exp(M_s - M_{\text{final}})$.
38. &emsp;&emsp;&emsp;Combine the output vectors with proper scaling: $O_{\text{final}} \leftarrow O_{\text{final}} \cdot \alpha + O_s \cdot \beta$.
39. &emsp;&emsp;&emsp;Combine the sum values: $L_{\text{final}} \leftarrow L_{\text{final}} \cdot \alpha + L_s \cdot \beta$.
40. &emsp;&emsp;**end for**
41. &emsp;&emsp;Normalize the final output vector: $O_{\text{final}} \leftarrow O_{\text{final}} / L_{\text{final}}$.
42. &emsp;&emsp;Write the final $O_{\text{final}}$ back to its designated position in HBM.
43. &emsp;**end Parallelize**
44. **end if**
