# Head-wise adaptive sparse attention

The authoritative source is the supplied HASTE.pdf, arXiv:2605.14513v1,
14 May 2026. The later revision uses the title HEART. This implementation follows
the supplied HASTE version and adapts its Wan2.1 experiments to Wan2.2 Animate 2.

## Sparse attention

For query i and key j, retained attention is

\[
A_{ij}=\frac{\exp(q_i^T k_j/\sqrt d)\,M_{ij}G_{ij}}
{\sum_r\exp(q_i^T k_r/\sqrt d)\,M_{ir}G_{ir}},\qquad Y=AV.
\]

V contains value vectors and G is the structural visibility indicator. The sparse
mask M is selected from current activations without changing pretrained parameters.

XAttention concatenates reversed query positions and forward key positions within
each stride group. Its reduced logits are the average antidiagonal dot product,
scaled by the inverse square root of the head dimension. Softmax is evaluated over
reduced key groups, then summed into block importance. In the Animate 2 extension,
forbidden and padded antidiagonal pairs are excluded from the average.

SVG2 uses Euclidean clustering of Q and K, stable token permutations, and centroid
scores weighted by key cluster population. Initial centers are sampled from actual
tokens with a fixed clustering seed. Empty clusters retain their prior centers.
Warm starts use centers from the last refresh. Defaults are 300 query clusters,
1000 key clusters, 50 initial Lloyd iterations, and 2 update iterations, with counts
capped by sequence length. A minimum key-cluster fraction of 0.1 follows the public
SVG2 configuration. No dense warmup steps or early dense layers are added.

For masked rectangular attention, the SVG2 key weight becomes the number of
permitted query/key pairs in a cluster pair divided by the query cluster size.
On unrestricted attention this reduces to the original key cluster population.
Both backends select the shortest descending prefix reaching the top-p mass.
Ties use ascending cluster indices. Full retention selects every permitted pair,
including entries whose estimated probabilities underflow to zero.

Generation keys are visible to every query. Reference frame r is visible only to
generation frame r+1. Sparse selection includes both streams. If the selected
clusters contain no generation key, the highest scoring generation cluster is
added to ensure every query has a valid connection. Token-level visibility remains
enforced inside selected blocks, including after SVG2 permutation.

The processor supports the pinned checkpoint's complete segment grid. It rejects
incompatible cropped grids instead of silently changing upstream padding behavior.
The mathematical and CUDA attention primitives separately support rectangular
sequences, partial blocks, and references outside the query frame range.
Reference extraction and text/image cross-attention remain upstream. The original
feedforward projections, normalization, positional embeddings, and output biases
are preserved.

## Temporal mask reuse

HASTE Equation 5 uses means over valid token positions:

\[
\bar q_t={1\over N_q}\sum_i q_{t,i},\quad
\bar k_t={1\over N_k}\sum_j k_{t,j},\quad
D_t=\|\bar q_t-\bar q_a\|_1+\|\bar k_t-\bar k_a\|_1.
\]

The anchor a is the last mask refresh, as in Algorithm 1. A missing cache or drift
strictly greater than delta refreshes that head. Equality reuses the mask. Reuse
leaves the anchor unchanged and skips scoring, selection, and semantic clustering.
A changed threshold also forces refresh. The no-reuse mode refreshes every call,
including identical inputs.

SVG2 reuses its complete anchor partition: query labels, key labels, permutations,
cluster offsets, and mask. Current QKV values are gathered using those cached token
indices. Reusing only a cluster mask with newly permuted tokens would be incorrect.

Defaults are delta 30 for XAttention and 8 for SVG2, as in Section 5.1. The
Section 4.2 layer-level gating heuristic is available through the gate setting:
when the fraction of heads marked for refresh within a layer is below the lower
bound, the whole layer reuses; above the upper bound, the whole layer refreshes;
between the bounds the head-wise decisions stand. Gating applies to drift
decisions only; first calls, no-reuse mode, and threshold changes always refresh.
The paper gives no bound values, so gating is disabled unless configured. The
refresh decision and anchor update run as a few small device operations shared by
both engines without host synchronization. These are experimental settings, not
validated Wan2.2 quality guarantees. The paper's full-token drift bound has extra
assumptions and does not establish a guarantee for this pooled statistic or the
Animate 2 extension.

State is separate for samples, heads, layers, and guidance branches. Reference
extraction clears both branches at a segment boundary. Reference identity, device,
dtype, geometry, and configuration changes invalidate cached state. Removing the
patch restores the exact original processors and releases device state.

## Error-guided budgeted calibration

Algorithm 2 evaluates one sparse head at a time against cached dense denoising
velocities. Each head receives one reproducibly assigned development sample and
one timestep from each of four equal trajectory intervals. Candidates 0.85, 0.90,
and 0.95 use the same sampled calls. Calibration covers the first segment of each
assigned video; this bounded sampling choice is saved with the measurements.

The cache stores transformer inputs, conditioning, reference projections, geometry,
timesteps, and dense outputs. Replaying a candidate starts from the saved dense
latent, rather than a latent modified by an earlier sparse trajectory. Only the
chosen head's projected output difference is added to the upstream dense result.
Its dense comparison uses the same token layout and accumulation path. Other
heads remain on the upstream path. Conditional and unconditional branches are
measured separately; Animate 2's skipped unconditional block 9 is excluded.

Equations 9 to 11 define the velocity error and weighted spectral energy:

\[
\epsilon_h=y_h^{\rm sparse}-y^{\rm dense},\qquad
r_{hq}={\sum_{\omega\in\Omega_q}|\mathcal F(\epsilon_h)_\omega|^2
\over \sum_\omega|\mathcal F(y^{\rm dense})_\omega|^2+\varepsilon},
\qquad e_h=\sum_q w_qr_{hq}.
\]

FFT dimensions are latent time, height, and width, with orthonormal normalization.
The four weights for LL, LH, HL, and HH are (1, 0.5, 0.01, 0.01). Low frequencies
satisfy absolute fftfreq less than 0.25 cycles per sample. Spatial low requires both
spatial axes to satisfy the cutoff. These explicit boundaries are an implementation
choice because the supplied paper does not specify them. No FFT shift is needed
when bands are constructed from signed frequencies directly.

Equation 7 selects one candidate per active head:

\[
\min_x\sum_{h,k}E_{hk}x_{hk},\quad
\sum_kx_{hk}=1,\quad {1\over H}\sum_{h,k}S_{hk}x_{hk}\ge S_{\min},
\quad x_{hk}\in\{0,1\}.
\]

SciPy MILP solves each guidance branch separately. The budget is the measured mean
sparsity of the shared 0.90 setting for that backend. Empty or infeasible problems
fail explicitly. Feasible time-limited solutions retain their nonzero optimality
gap and are not reported as proven optimal. The additive objective neglects
interactions between heads and layers, as discussed in Appendix B.

Sparsity counts only originally permitted token pairs. Pair counts account for
partial blocks and variable cluster populations. Retained block counts are also
reported; they must not be interpreted as token-pair density or measured speed.

## CUDA and measurement

BF16 QKV and output tensors use FP32 dot-product accumulation and online softmax.
Cluster centers, pooled features, and score reductions use FP32. CUDA cluster
assignment uses TF32x3 for FP32 centroid products; it does not reduce model weights
to TF32. CPU references use independent dense formulas for small tensors.

XAttention scratch scores are limited to four query blocks against the reduced
key sequence. Sparse multiplication uses 32-query by 32-key tiles and visits
selected key clusters. SVG2 assignment and centroid updates use bounded tiles;
no full token attention matrix is constructed in the CUDA path. CUDA refresh
flags guard scoring, clustering, layout construction, and selection on device.
No per-head host decision or synchronization is required.

The kernels are research implementations. CUDA compilation, correctness, memory
feasibility, and performance have not been verified on H100. Head-wise refresh
reduces work but still incurs launch and metadata overhead. Positive speedup and
preserved perceptual quality are not established by CPU tests.
