# Linear extension of HASTE

The supplied paper is Meiner et al., arXiv:2606.30516, rather than the unrelated
HASTE sparse attention method. Its CNN experiments do not establish performance
for Wan. Section 5.2 motivates embedding channel compression in linear layers and
explicitly identifies the overhead risk.

## Window and hash

Let X have shape T by C within one sample's contiguous token window. Each input
channel is represented by its T activation values. Center only the hashing input:

\[
Z_{tc}=X_{tc}-\frac{1}{C}\sum_{j=1}^{C}X_{tj}.
\]

For L hyperplanes R of shape T by L, use the sign convention from Equations 3 to 5:

\[
h_c=\sum_{l=0}^{L-1}2^l\,\mathbf{1}\left[\sum_t Z_{tc}R_{tl}>0\right].
\]

Gaussian projections follow Section 3.1. Ternary projections follow Section 3.4:
P(R=0)=s and P(R=1)=P(R=−1)=(1−s)/2. A projection is sampled once for each selected
layer and remains fixed. A final short window uses the corresponding prefix of
the projection matrix. Samples and windows never share bucket membership.

The implementation deliberately retains possible zero ternary hyperplanes rather
than conditioning their distribution by resampling. Zero dot products map to bit
zero, matching the paper's strict positive test.

## Compressed linear map

For an occupied bucket B, Equation 9 extends to a linear projection as:

\[
\bar X_{tB}=\frac{1}{|B|}\sum_{c\in B}X_{tc},\qquad
\widetilde W_{oB}=\sum_{c\in B}W_{oc},\qquad
\widehat Y_{to}=\sum_B\bar X_{tB}\widetilde W_{oB}+b_o.
\]

The input means use uncentered X. Bias is added once, after the reduced product.
The exact residual is:

\[
Y_{to}-\widehat Y_{to}
=\sum_B\sum_{c\in B}(X_{tc}-\bar X_{tB})W_{oc}.
\]

Singleton buckets and buckets containing identical channel vectors therefore
recover the dense linear result up to floating point roundoff. Similar direction
alone does not guarantee a small residual when channel magnitudes differ.

## Cost and precision

The dense product uses approximately 2TCO floating point operations. With K
occupied buckets, the reduced product uses approximately 2TKO operations, but
weight merging still reads CO weights per window. Hashing, sorting, compaction,
activation averaging, weight merging, allocations, and launches must all be
included in latency comparisons. Reduced product FLOPs are not a speedup claim.

CUDA reductions and merged operands use FP32. The product uses TF32x3 with FP32
accumulation and casts the final result to BF16 for model execution. The reference
uses FP32 matrix multiplication. Hash membership can change near zero when
reduction order changes; kernel validation checks membership separately from
output tolerance.

Only occupied buckets are compacted. Storage scales with C rather than 2 raised
to L. With output tile U, principal merge workspace contains T times min(C,2^L)
activation values and U times min(C,2^L) weight values, plus linear size indexing
buffers. Pretrained weights are never summed in place.

Both linear projections are independently eligible. GELU, attention, normalization,
conditioning, and output dimensions remain unchanged. Selected feedforward layers
are compressed during both reference extraction and denoising; the attention cache
mechanism is untouched, although its values may change as a consequence of earlier
feedforward approximation.

## Measurement protocol

Use paired samples with identical generation seeds and a separate hashing seed.
Warmup records include first use compilation and are excluded from steady timings.
Preview encoding and metrics run outside generation timings. Report median repeated
wall time, individual samples, CUDA intervals for reference extraction and denoising,
and peak allocated and reserved device memory.

Bucket instrumentation runs in a separate diagnostic generation. PSNR uses global
video MSE with data range one. SSIM and AlexNet LPIPS are averaged over frames.
Temporal error is the mean absolute difference between the two videos' adjacent
frame differences. Metrics use lossless arrays, not encoded preview videos.

Choose an explicit configuration from development comparisons before evaluating
held out videos. Summaries compare Pareto candidates only within matching input,
sampling, hardware, and implementation cohorts. No universal perceptual threshold
or guaranteed speedup is assumed.
