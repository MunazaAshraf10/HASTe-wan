# HASTE for Wan Animate 2

This repository implements HASTE, Training Free Video Diffusion Acceleration via
Head Wise Adaptive Sparse Attention, for Wan2.2 Animate 2. The implementation
targets the quadratic cost of spatiotemporal self attention during denoising while
leaving pretrained weights, feedforward layers, cross attention, normalization,
rotary embeddings, and reference extraction unchanged.

The supplied [HASTE paper](HASTE.pdf) validates the method on Wan2.1. This work
extends it to the rectangular reference attention used by Wan2.2 Animate 2.
Generation tokens can attend to every generation key, while each frame can attend
only to its corresponding reference frame. Sparse selection includes both key
streams and preserves this visibility constraint inside every selected block.

## Implemented method

Temporal Mask Reuse follows Equation 5 and Algorithm 1. Every head stores the
mean query and key vectors from its latest mask refresh. The current drift is

\[
D_t=\lVert\bar q_t-\bar q_a\rVert_1+\lVert\bar k_t-\bar k_a\rVert_1.
\]

A head refreshes only when its drift exceeds the configured threshold. Reused
heads retain their anchor, mask, and sparse layout while applying the current
query, key, and value tensors. State is isolated by sample, layer, attention head,
guidance branch, reference cache, segment geometry, device, and data type.

XAttention uses inverse stride antidiagonal scoring followed by block level top p
selection. Forbidden reference pairs are excluded before score normalization.
Scoring workspace is tiled across query blocks so memory does not scale as a full
token attention matrix.

SVG2 performs Euclidean query and key clustering, centroid population weighted
scoring, stable semantic permutation, and top p cluster selection. Mask reuse
preserves labels, permutations, offsets, centroids, and selected cluster pairs as
one consistent anchor state. Empty clusters retain their preceding centers.

Error Guided Budgeted Calibration follows Equations 7 and 9 through 11. It
measures one sparse head at a time against cached dense denoising velocities,
computes weighted four band temporal and spatial FFT error, and solves the global
sparsity budget with integer linear programming. Conditional and unconditional
guidance branches receive separate threshold tables. Calibration inputs, seeds,
model revision, environment, measurements, solver status, and selected thresholds
are stored with the result.

## CUDA implementation

Triton kernels implement cluster assignment, centroid updates, stable layout
construction, structural pair histograms, XAttention scoring, probability
normalization, top p selection, and sparse attention. The attention kernel uses
online softmax across selected key tiles and never constructs the complete token
score matrix.

Model inputs and outputs use BF16. Dot products, softmax statistics, cluster
centers, pooled features, drift calculations, and numerical reductions use FP32.
This keeps model storage and execution in BF16 while using FP32 where accumulated
rounding error would otherwise affect masks or attention probabilities.

The target environment is Python 3.14.6 on Linux x86_64 with an NVIDIA H100 and a
driver compatible with CUDA 13. PyTorch, Triton, Diffusers, model revision, and
all dependencies are pinned by the project lockfile.

## Correctness evidence

The independent PyTorch reference covers rectangular masked attention, partial
blocks, top p selection, antidiagonal scoring, semantic clustering, stable
permutation, token pair accounting, and HASTE anchor behavior. Mathematical tests
verify full retention against dense attention, sparse output against an explicit
mask oracle, four band FFT energy, and the calibration solver against exhaustive
enumeration.

Integration tests verify reversible processor installation, exact restoration of
the original Wan processors, unchanged parameters and reference cache tensors,
guidance branch isolation, segment invalidation, and isolated head calibration.
CUDA acceptance tests compare both sparse backends with the mathematical reference
at representative dimensions and check mixed refresh and reuse behavior.

Local CPU validation currently passes 42 tests together with formatting, linting,
and static type checks. Ten CUDA or external metric tests are skipped when their
required hardware or public metric weights are unavailable. H100 kernel
correctness, full model memory use, visual quality, and measured acceleration must
be established on the target GPU before performance claims are made.
