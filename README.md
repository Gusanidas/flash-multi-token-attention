Implementations of multi token attention in CUDA and Triton.

- (Paper)[https://arxiv.org/abs/2504.00927]
- (Original implementation)[https://github.com/facebookresearch/RAM/tree/main/projects/mta]


## Conv Attention ##

In this variant, there is only a convolution before the softmax. 

A = Softmax Conv2dθ (Aˆ)

aij = Softmax   cq−1 ∑ i ′=0 ⌈ck/2⌉−1 ∑ j ′=−⌊ck/2⌋ 1i≥j−j ′ θi ′ ,j ′ qi−i ′ k ⊤ j−j ′ / √ d

The output is:

Out_j,h = ∑_i a_j,i * V_i,

The implementation fuses all these operations in a single CUDA kernel, using the main ideas of tiling and online softmax from Flash Attention.

Because of the convolution, the tiles overlap each other, and there are some values of the Q*K^T matrix that are recomputed.


This kernel achieves some speedup compared with the pytorch implementation:
![speed-up1](benchmark_seq_len_speedup.png)

Because of the unavoidable extra computations involved in the convolution, it is slower than standard flash attention in the training workload:

![fwd_conv](benchmark_seq_len_backward_times.png)
![bwd_conv](benchmark_seq_len_forward_times.png)

However, during inference, memory bandwith is the bottleneck and increasing arithmetic intensity doesnt affect the performance as much:

![fwd_inference](fwd_inference_speedup.png)

