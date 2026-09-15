# Experiments

Install the pinned Python environment with uv sync on Linux. The default TOML
selects XAttention with TMR at 480 by 320 pixels. Change the backend to svg2 for
the second implementation. The reference engine is restricted to small tensors;
CUDA is required for generation and calibration.

The CLI provides generate, benchmark, calibrate, kernels, compare, and summarize.
Each of the first four takes a TOML configuration path. Summarize takes the
results directory. Compare takes two saved generations (.npy or .mp4) and an
optional output directory. Generate accepts the baseline flag; benchmark accepts
the sweep flag. Use the CLI help for argument spelling.

The quick TOML runs one warmup and one timed generation per mode with the trace
and counter passes disabled, so a first paired look costs four generations. The
default TOML keeps three timed repeats and full diagnostics.

Copy the example JSONL manifest and supply real reference images, driving videos,
prompts, generation seeds, and dev/eval split labels. Input paths are relative to
the manifest. Content-identical driving videos cannot cross the split boundary.
Generation outputs are saved as lossless NumPy arrays with MP4 previews, a PNG
contact sheet, and PNG files for four evenly spaced frames. A paired benchmark
also writes comparison.png with baseline, HASTE, and amplified absolute
difference rows, captioned with per-frame SSIM, and prints the timing and
quality summary.

Run calibrate separately for each backend on the development split. It writes
results/calibration/<configuration id>/table.json and per-GPU replay data. Offline
replay tensors can require substantial disk space; reference projections are shared
across sampled steps within each branch. Assign the resulting table path to the
haste table setting before choosing ebc or full mode. TMR and sparse modes do not
require calibration. Default mode is tmr so the initial configuration is runnable
before a table exists.

A benchmark sweep compares sparse, tmr, ebc, and full for both backends. Provide a
table path containing the {backend} placeholder, with compatible tables placed at
the corresponding xattention and svg2 paths. Tables are validated before generation.
Every comparison includes its own paired upstream dense generation on the same
worker. Configurations are restricted to development sweeps; freeze settings and
select the eval split for final measurement. Calibration input hashes are checked
against evaluation videos even when a different manifest is supplied.

One process owns each selected GPU. Workers load independent model copies; this
is experiment parallelism, not model sharding. Calibration assigns complete head
candidate sets to workers. Generation assigns complete baseline/candidate pairs.
Failed runs retain their status and error record. Existing output directories are
not overwritten.

Report compilation-inclusive first calls, warmup samples, synchronized repeated
latency, and peak allocated and reserved memory. Compilation and warmup are outside
the reported steady median. Transformer CUDA events separate reference extraction
and denoising; total generation includes preprocessing inside the pipeline and
output transfer, but excludes preview encoding and quality metrics.

Attention counters and phase events are collected in a separate diagnostic run.
They cover drift, scoring, clustering, permutation, selection, and attention.
Kernel benchmarks use fixed QKV inputs to isolate mask refresh and reuse behavior;
these timings do not represent a changing denoising trajectory.

PSNR, SSIM, AlexNet LPIPS, and temporal difference error compare paired lossless
outputs. Compilation diagnostics and counter collection are excluded from paired
wall-time speedups. Summaries compare development frontiers only across matching
input, model, source, and hardware cohorts. Full-model quality, GPU numerical
agreement, and H100 performance must be measured before making acceleration claims.
