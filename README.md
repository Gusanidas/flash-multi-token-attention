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
6. &emsp;$I_s = (i-1) \cdot B_r - P_q$. $I_e = I_s + B$.
7. &emsp;// Define indices for the effective output region (size $B_r$)
8. &emsp;$I_s^{\text{eff}} = (i-1) \cdot B_r$. $I_e^{\text{eff}} = I_s^{\text{eff}} + B_r$.
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
20. &emsp;&emsp;On chip, compute local stats $\tilde{m}_{ij}, \tilde{P}_{ij}, \tilde{\ell}_{ij}$.
21. &emsp;&emsp;// Online Softmax Update
22. &emsp;&emsp; $m_i^{\text{new}} = \max(m_i, \tilde{m}_{ij})$
23. &emsp;&emsp;Calculate rescaling factors: $\alpha = e^{m_i - m_i^{\text{new}}}$, $\beta = e^{\tilde{m}_{ij} - m_i^{\text{new}}}$.
24. &emsp;&emsp;Update $\ell_i \leftarrow \alpha \cdot \ell_i + \beta \cdot \tilde{\ell}_{ij}$.
25. &emsp;&emsp;// Update Output (Read-Modify-Write HBM)
26. &emsp;&emsp;Load the full tile $O[I_s:I_e]$ from HBM to SRAM.
27. &emsp;&emsp;On chip, $O[I_s:I_e] \leftarrow \alpha \cdot O[I_s:I_e] + (\beta \cdot \tilde{P}_{ij})V[J_s:J_e]$.
28. &emsp;&emsp;Write back only the effective region $O[I_s^{\text{eff}}:I_e^{\text{eff}}]$ to HBM.
&emsp;&emsp;&emsp;&emsp;&emsp;(Corresponds to the slice $[P_q:B]$ of the SRAM buffer).
29. &emsp;&emsp; $m_i \leftarrow m_i^{\text{new}}$
30. &emsp;**end for**
31. &emsp;// Finalization (Load, Normalize, Write effective region)
32. &emsp;Load the accumulated $O[I_s^{\text{eff}}:I_e^{\text{eff}}]$ from HBM.
33. &emsp;Normalize using the corresponding slice of the local statistics (held in SRAM):
&emsp;&emsp; $O[I_s^{\text{eff}}:I_e^{\text{eff}}] \leftarrow O[I_s^{\text{eff}}:I_e^{\text{eff}}] / \ell_i[P_q:B]$.
34. &emsp;Write normalized $O[I_s^{\text{eff}}:I_e^{\text{eff}}]$ to HBM.
35. &emsp;Write $m_i[P_q:B]$ to $M[I_s^{\text{eff}}:I_e^{\text{eff}}]$ and $\ell_i[P_q:B]$ to $L[I_s^{\text{eff}}:I_e^{\text{eff}}]$ in HBM.
36. **end Parallelize**
