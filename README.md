# HASTE for Wan Animate 2

This repository implements the supplied [HASTE paper](HASTE.pdf), Training Free
Video Diffusion Acceleration via Head Wise Adaptive Sparse Attention, for Wan2.2
Animate 2. It reduces denoising attention cost without changing pretrained
weights, feedforward layers, cross attention, normalization, rotary embeddings,
or reference extraction.

HASTE is implemented over XAttention and SVG2. Sparse selection covers generation
and reference keys while preserving Animate 2 frame visibility. The paper validates
Wan2.1; applying it to Wan2.2 Animate 2 is an extension that requires independent
GPU evaluation.

## Method

Temporal Mask Reuse stores each head's query and key means at its latest mask
refresh. Following Equation 5 and Algorithm 1, the current drift is

\[
D_t=\lVert\bar q_t-\bar q_a\rVert_1+\lVert\bar k_t-\bar k_a\rVert_1.
\]

A head refreshes when its drift exceeds the threshold. Otherwise it reuses the
anchor mask and sparse layout with current query, key, and value tensors. State is
isolated by sample, layer, head, guidance branch, reference cache, and segment.

XAttention uses inverse stride antidiagonal block scoring. SVG2 uses Euclidean
clustering, centroid weighted scoring, stable semantic permutation, and top p
cluster selection. Reused SVG2 state keeps labels, permutations, offsets,
centroids, and masks consistent.

Error Guided Budgeted Calibration measures isolated head errors against cached
dense denoising velocities. Equations 7 and 9 through 11 define a weighted four
band temporal and spatial FFT objective and an integer program that assigns one
top p threshold per head under a global sparsity budget. Guidance branches are
calibrated separately.

## CUDA implementation

Triton kernels implement scoring, clustering, stable permutation, top p selection,
mask reuse, and tiled sparse attention with online softmax. The CUDA path does not
construct a full token score matrix. Model tensors use BF16; dot products, softmax,
cluster centers, drift, and reductions accumulate in FP32.

The target is Python 3.14.6 on Linux x86_64 with NVIDIA H100 and a driver compatible
with CUDA 13. Dependencies and the Wan model revision are pinned.

## Correctness evidence

The PyTorch reference verifies sparse attention against explicit masks, full
retention against dense attention, FFT energy, TMR anchor behavior, and calibration
against exhaustive search. Integration tests verify unchanged weights, reference
caches, branch isolation, segment invalidation, and exact processor restoration.

Local validation passes 42 tests plus formatting, linting, and type checks. H100
kernel correctness, full model memory use, visual quality, and speed remain to be
measured before performance claims are made.
