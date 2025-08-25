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


