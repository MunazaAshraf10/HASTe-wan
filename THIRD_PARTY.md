# Algorithm and implementation references

HASTE: Training-Free Video Diffusion Acceleration via Head-Wise Adaptive Sparse
Attention, Zheng et al., arXiv:2605.14513v1. The supplied HASTE.pdf is retained as
the methodological reference. Paper copyright remains with its authors.

[XAttention](https://github.com/mit-han-lab/x-attention/tree/e37988770b9d1bebd489eba011d615f35587ba08),
Xu et al., ICML 2025. The antidiagonal estimator is independently implemented from
the published algorithm and checked against its public inverse-stride formulation.
No upstream XAttention source files are redistributed.

[Sparse VideoGen2](https://github.com/svg-project/Sparse-VideoGen/tree/f89aedaf169ac2ae5b186bda674e53c3dc08c476),
Yang et al., NeurIPS 2025. Euclidean clustering, centroid population weighting,
semantic permutation, and minimum cluster retention follow its public algorithm.
The CUDA and PyTorch implementations here are adapted for rectangular masked
attention and head-wise reuse. The upstream project uses the Apache 2.0 license,
reproduced in licenses/Apache-2.0.txt.

[Diffusers 0.40.0](https://github.com/huggingface/diffusers/tree/v0.40.0),
Copyright 2026 The HuggingFace Team. The Animate 2 processor adaptation follows
its QKV normalization, rotary embeddings, reference cache semantics, and output
projection flow. Diffusers uses the Apache 2.0 license, reproduced in
licenses/Apache-2.0.txt. Modified attention execution is provided in this project's
Wan integration; the upstream package is installed as a dependency.
