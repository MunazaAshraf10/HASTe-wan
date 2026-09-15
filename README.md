# HASTE for Wan Animate 2

This repository implements [HASTE](HASTE.pdf), Training Free Video Diffusion
Acceleration via Head Wise Adaptive Sparse Attention, for Wan2.2 Animate 2. It
accelerates denoising self attention without modifying pretrained weights, cross
attention, feedforward layers, normalization, rotary embeddings, or reference
extraction.

The paper validates Wan2.1. This code extends HASTE to Animate 2 rectangular
attention, where generation queries attend to generation keys and frame aligned
reference keys. Structural visibility is enforced during scoring and sparse
attention.

## Method

Temporal Mask Reuse stores FP32 token means for each attention head at the most
recent mask refresh. Equation 5 is implemented as

$$
D_t = \left\lVert \bar{q}_t - \bar{q}_a \right\rVert_1
    + \left\lVert \bar{k}_t - \bar{k}_a \right\rVert_1
$$

where a denotes the last refresh step. A head refreshes when D exceeds its drift
threshold. Otherwise the cached sparse layout is applied to current Q, K, and V.
Caches are isolated by sample, layer, head, guidance branch, reference cache, and
segment geometry.

XAttention uses inverse stride antidiagonal scoring followed by block top p
selection. SVG2 uses Euclidean Q and K clustering, centroid population weighting,
stable semantic permutation, and cluster top p selection. SVG2 reuse preserves
centroids, labels, permutations, offsets, and masks as one anchor state.

Error Guided Budgeted Calibration evaluates one sparse head at a time against
cached dense denoising velocities. A weighted four band 3D FFT objective measures
the output error. Integer linear programming then selects one threshold per head
under the measured global sparsity budget. Conditional and unconditional guidance
branches are calibrated independently.

## Code structure

| Component | Implementation |
| --- | --- |
| HASTE state and execution | [attention.py](src/haste/attention.py) |
| Animate 2 processor integration | [wan.py](src/haste/wan.py) |
| Mathematical reference | [reference.py](src/haste/reference.py) |
| Triton CUDA kernels | [kernels.py](src/haste/kernels.py) |
| EBC objective and solver | [calibration.py](src/haste/calibration.py) |
| Dense replay and calibration workers | [calibrate.py](src/haste/calibrate.py) |
| Generation and paired evaluation | [runner.py](src/haste/runner.py) |

The execution modes are sparse attention, sparse attention with TMR, sparse
attention with EBC, and the complete TMR plus EBC method. Processor installation
is reversible and restores the original Wan attention processors exactly.

## CUDA and validation

Triton kernels implement scoring, clustering, permutation, top p selection, and
tiled sparse attention with online softmax. The optimized path does not construct
the full token score matrix. Model tensors use BF16. Dot products, softmax state,
cluster centers, drift, and reductions use FP32.

The target is Python 3.14.6, Linux x86_64, NVIDIA H100, and a driver compatible
with CUDA 13. Dependencies and the Wan model revision are pinned.

Local validation passes 42 tests, formatting, linting, and type checking. Tests
cover dense equivalence, explicit sparse mask agreement, TMR anchor behavior,
reference visibility, FFT energy, calibration optimality, cache isolation,
unchanged weights, and processor restoration. H100 correctness, memory use,
quality, and speed remain unverified until GPU execution.
