# HASTE for Wan Animate 2

Video diffusion repeatedly evaluates large feedforward projections during
denoising. These dense operations process every input channel, even when channel
activations contain redundant information. This project investigates reducing
that computation in Wan Animate 2 without retraining its pretrained weights.

## Method

[HASTE](HASTE.pdf), Hashing for Tractable Efficiency, uses locality sensitive
hashing to identify and merge similar activation channels in convolutional neural
networks. This repository extends that principle to transformer feedforward layers.

Within each token window, random projections group channels by their centered
activation patterns. Channels in each group are averaged, and the corresponding
weight columns are summed. The resulting linear operation uses fewer input
channels while retaining the original output dimensions.

For T tokens, C input channels, and O output channels, the dense product requires
approximately 2TCO operations. With K occupied groups, the compressed product
requires approximately 2TKO operations. Practical acceleration depends on these
savings exceeding the cost of hashing, grouping, and weight aggregation.

## Runtime

Python 3.14.6, Linux x86_64, NVIDIA H100, and an NVIDIA driver compatible with
CUDA 13. The implementation uses PyTorch and Triton with BF16 model execution
and FP32 reductions.
