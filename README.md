# HASTE for Wan Animate 2

This package extends the supplied HASTE paper's channel compression to the two
feedforward projections in Wan Animate 2. It is an experimental transformer
extension, not a reproduction of the paper's CNN accuracy or speed results.

The method hashes channel activation vectors within each token window, averages
channels sharing a hash, and sums their corresponding weight columns. The original
weights and output dimensions are preserved. Attention and the reference cache
implementation remain upstream Diffusers components.

## Environment

Use Python 3.14.6 and uv sync. The lock records the resolved dependencies. Linux
x86_64 with an NVIDIA H100 is the execution target. On macOS, installation supports
reference tests and static checks; generation requires CUDA.

The Linux lock includes CUDA 13 runtime libraries through PyTorch. The H100 host
needs a compatible NVIDIA driver. The current stable Diffusers 0.40.0 provides
WanAnimate2ModularPipeline; the package uses this public API instead of the older
model card's pipeline class. Checkpoint revision is fixed in the configuration.

Run uv run pytest, uv run ruff check ., uv run ruff format --check ., and uv run ty
check for local validation. GPU tests carry the cuda marker and are skipped when
CUDA is unavailable. The H100 test command is uv run pytest -m cuda.

Keep imports at the top of each Python module and use unquoted type annotations.
Place concise documentation on functions and classes rather than at the top of
files. Optional Triton imports belong in the backend module's import section.

## Experiments

Edit experiments/default.toml and provide a JSONL manifest. Each row contains name,
image, video, prompt, seed, and split. Paths are relative to the manifest. Split is
dev or eval. Driving video content must not overlap between the two splits.

Run uv run haste generate experiments/default.toml for compressed generation.
Add --baseline for exact upstream feedforward layers.

Run uv run haste benchmark experiments/default.toml for paired generation and
quality measurements. Add --sweep for the predefined development grid. Multiple
devices run independent experiment pairs, with one process per GPU.

Generation runs once. Benchmarks run the configured warmups and repetitions plus
separate tracing and bucket diagnostic passes. Existing artifact directories are
never overwritten. Use a new output directory for a rerun. The full development
sweep contains 75 configurations per manifest sample and can be expensive.

Configuration manifest and output paths are relative to the working directory.
The model height and width specify target area; upstream preprocessing preserves
the reference image's aspect ratio. Frames specifies segment length, not a limit
on the full driving video's length. The whole input video is processed.

Run uv run haste kernels experiments/default.toml for component timings at Wan
feedforward dimensions. Run uv run haste summarize results for aggregate reports
and the development Pareto frontier. Evaluation runs use explicitly selected
configuration files; parameter sweeps are restricted to development data.

The output includes lossless NumPy frames for scoring, preview videos, timing
samples, warmup costs, peak memory, bucket diagnostics, input hashes, dependency
versions, separate generation and hashing seeds, and configuration provenance.

The first LPIPS evaluation downloads torchvision's AlexNet weights. Scoring uses
the current TorchMetrics LPIPS interface and runs outside generation timing.
The mathematical specification and measurement definitions are in docs/method.md.

GitHub validation runs reference tests and static checks. The optional manual CUDA
job requires a registered runner with the h100 label; no GPU results are assumed.

## Interpretation

Section 5.2 of HASTE identifies linear layers as a possible extension and warns
that hashing and merging overhead can outweigh arithmetic savings. Report actual
wall time, including that overhead. Similarity to dense output does not establish
absolute video quality. No H100 speedup, memory fit, or numerical result has been
verified on the development machine.

The CUDA path uses device sorting followed by Triton compaction and merge kernels.
Merged operands use FP32; products use TF32x3 tensor core multiplication with FP32
accumulation and are cast to the model dtype. Workspace is bounded by token windows
and output channel tiles. Dynamic compression runs outside compiled transformer
graphs, while upstream attention remains compiled.

## References

Meiner, Mehnert, and Condurache. HASTE: A Framework for Training-Free, Dynamic, and
Steerable Compression of Pre-Trained Convolutional Neural Networks. 2026.
https://arxiv.org/abs/2606.30516

The supplied HASTE.pdf is the method reference. Equation 9 defines channel
averaging and filter summation. Equations 3 through 5 define the hash; Section 3.4
defines ternary projections. The implementation hashes centered vectors and merges
uncentered values. Token windows, linear projection selection, and CUDA execution
are extensions developed here.

Wan Animate 2 public model:
https://huggingface.co/Wan-AI/Wan2.2-Animate-2-14B-Diffusers

Diffusers public implementation and license:
https://github.com/huggingface/diffusers

No private ml-api source is included. Third party packages retain their licenses.
