# HASTE for Wan Animate 2

Video diffusion repeatedly computes attention over large spatiotemporal token
sequences. Sparse attention reduces this quadratic cost, but rebuilding sparse
masks at every denoising step adds overhead, and a shared sparsity threshold
ignores differences between attention heads.

[HASTE](HASTE.pdf), Training-Free Video Diffusion Acceleration via Head-Wise
Adaptive Sparse Attention, addresses these problems through temporal mask reuse
and error-guided allocation of per-head attention budgets. This project adapts
that method to Wan2.2 Animate 2 with XAttention and SVG2, including its reference
video attention. The supplied paper evaluates Wan2.1; Wan2.2 performance requires
independent measurement.

Python 3.14.6, Linux x86_64, NVIDIA H100, and an NVIDIA driver compatible with
CUDA 13. PyTorch and Triton use BF16 model tensors with FP32 accumulation for
attention, softmax, and numerical reductions.

## Baseline versus HASTE

Install on the GPU host with uv sync, copy experiments/manifest.example.jsonl to
experiments/manifest.jsonl, and point it at a reference image and driving video.
Then, from the repository root:

    uv run haste benchmark experiments/quick.toml

This loads the model once, generates the dense baseline, installs HASTE on the
same model, generates again, and prints the median latency of each, the speedup,
and SSIM, PSNR, and LPIPS between the two videos. Under
results/benchmark/<configuration id>/<sample name>/ it writes baseline and haste
MP4 previews, lossless .npy arrays, PNG frames, and comparison.png with the two
videos side by side over an amplified difference row.

Baseline and HASTE can also be generated as separate jobs and scored afterwards:

    uv run haste generate experiments/quick.toml --baseline
    uv run haste generate experiments/quick.toml
    uv run haste compare results/baseline/<id>/<name>/baseline.npy \
        results/generate/<id>/<name>/haste.npy

Timings from generate are single calls that include compilation; benchmark
reports warmed, repeated medians. For the full method, run calibrate on the
development split, set the resulting table path in the TOML, and choose mode
full (TMR with calibrated per-head thresholds). Mode tmr needs no calibration.
