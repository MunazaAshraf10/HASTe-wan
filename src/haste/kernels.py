'''CUDA kernels for windowed HASTE channel compression.

Hash sorting uses PyTorch's device implementation. Triton compacts the sorted
codes, merges occupied groups, and evaluates the reduced matrix product.
All reductions accumulate in FP32; merged operands remain FP32 until the dot.
'''

import torch
import triton as tr
import triton.language as tl
from torch import Tensor


@tr.jit
def center_kernel(
    X,
    Mean,
    T: tl.constexpr,
    C: tl.constexpr,
    SX: tl.constexpr,
    SC: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    value = tl.load(X + t * SX + c * SC, c < C, 0).to(tl.float32)
    tl.store(Mean + t, tl.sum(value, 0) / C)


@tr.jit
def hash_kernel(
    X,
    Mean,
    Planes,
    Codes,
    T: tl.constexpr,
    C: tl.constexpr,
    L: tl.constexpr,
    SX: tl.constexpr,
    SC: tl.constexpr,
    WINDOW: tl.constexpr,
    CHANNELS: tl.constexpr,
    TERNARY: tl.constexpr,
):
    c = tl.program_id(0) * CHANNELS + tl.arange(0, CHANNELS)
    t = tl.arange(0, WINDOW)
    value = tl.load(
        X + t[:, None] * SX + c[None, :] * SC, (t[:, None] < T) & (c[None, :] < C), 0
    ).to(tl.float32)
    mean = tl.load(Mean + t, t < T, 0)
    value = value - mean[:, None]
    code = tl.full((CHANNELS,), 0, tl.int32)
    for bit in range(L):
        plane = tl.load(Planes + t * L + bit, t < T, 0)
        if TERNARY:
            signed = tl.where(plane[:, None] > 0, value, tl.where(plane[:, None] < 0, -value, 0.0))
            score = tl.sum(signed, 0)
        else:
            score = tl.sum(value * plane[:, None], 0)
        code = code | ((score > 0).to(tl.int32) << bit)
    tl.store(Codes + c, code, c < C)


@tr.jit
def compact_kernel(Codes, Offsets, Count, C: tl.constexpr, BLOCK: tl.constexpr):
    c = tl.arange(0, BLOCK)
    code = tl.load(Codes + c, c < C, -1)
    prev = tl.load(Codes + c - 1, (c > 0) & (c < C), -2)
    start = (c < C) & ((c == 0) | (code != prev))
    group = tl.cumsum(start.to(tl.int32), 0) - 1
    tl.store(Offsets + group, c, start)
    count = tl.sum(start.to(tl.int32), 0)
    tl.store(Offsets + count, C)
    tl.store(Count, count)


@tr.jit
def value_kernel(
    X,
    Order,
    Offsets,
    Count,
    Values,
    T: tl.constexpr,
    C: tl.constexpr,
    SX: tl.constexpr,
    SC: tl.constexpr,
    CAP: tl.constexpr,
    WINDOW: tl.constexpr,
):
    group = tl.program_id(0)
    count = tl.load(Count)
    if group < count:
        lo = tl.load(Offsets + group)
        hi = tl.load(Offsets + group + 1)
        t = tl.arange(0, WINDOW)
        lane = tl.arange(0, 32)
        total = tl.full((WINDOW, 32), 0, tl.float32)
        for pos in range(lo, hi, 32):
            loc = pos + lane
            c = tl.load(Order + loc, loc < hi, 0)
            value = tl.load(
                X + t[:, None] * SX + c[None, :] * SC, (t[:, None] < T) & (loc[None, :] < hi), 0
            ).to(tl.float32)
            total += value
        avg = tl.sum(total, 1) / (hi - lo)
        tl.store(Values + t * CAP + group, avg, t < T)


@tr.jit
def weight_kernel(
    W,
    Order,
    Offsets,
    Count,
    Filters,
    C: tl.constexpr,
    OUT: tl.constexpr,
    BASE: tl.constexpr,
    ROWS: tl.constexpr,
    SW: tl.constexpr,
    SC: tl.constexpr,
    CAP: tl.constexpr,
):
    group = tl.program_id(0)
    row = tl.program_id(1) * 32 + tl.arange(0, 32)
    if group < tl.load(Count):
        lo = tl.load(Offsets + group)
        hi = tl.load(Offsets + group + 1)
        lane = tl.arange(0, 32)
        total = tl.full((32, 32), 0, tl.float32)
        for pos in range(lo, hi, 32):
            loc = pos + lane
            c = tl.load(Order + loc, loc < hi, 0)
            value = tl.load(
                W + (BASE + row[:, None]) * SW + c[None, :] * SC,
                (row[:, None] < ROWS) & (BASE + row[:, None] < OUT) & (loc[None, :] < hi),
                0,
            ).to(tl.float32)
            total += value
        value = tl.sum(total, 1)
        tl.store(Filters + row * CAP + group, value, row < ROWS)


@tr.jit
def product_kernel(
    Values,
    Filters,
    Count,
    Bias,
    Y,
    T: tl.constexpr,
    OUT: tl.constexpr,
    BASE: tl.constexpr,
    ROWS: tl.constexpr,
    CAP: tl.constexpr,
    SY: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    t = tl.program_id(0) * 32 + tl.arange(0, 32)
    o = tl.program_id(1) * 32 + tl.arange(0, 32)
    k = tl.arange(0, 32)
    count = tl.load(Count)
    total = tl.full((32, 32), 0, tl.float32)
    for start in range(0, count, 32):
        group = start + k
        a = tl.load(
            Values + t[:, None] * CAP + group[None, :],
            (t[:, None] < T) & (group[None, :] < count),
            0,
        )
        b = tl.load(
            Filters + o[None, :] * CAP + group[:, None],
            (o[None, :] < ROWS) & (group[:, None] < count),
            0,
        )
        total += tl.dot(a, b, input_precision='tf32x3')
    if HAS_BIAS:
        bias = tl.load(Bias + BASE + o, BASE + o < OUT, 0).to(tl.float32)
        total += bias[None, :]
    tl.store(
        Y + t[:, None] * SY + BASE + o[None, :],
        total,
        (t[:, None] < T) & (o[None, :] < ROWS) & (BASE + o[None, :] < OUT),
    )


def hashes(x: Tensor, planes: Tensor, ternary: bool = False) -> Tensor:
    '''Center and hash channels; ternary projections use signed additions.'''
    tokens, channels = x.shape
    mean = torch.empty(tokens, device=x.device, dtype=torch.float32)
    labels = torch.empty(channels, device=x.device, dtype=torch.int32)
    center_kernel[(tokens,)](x, mean, tokens, channels, *x.stride(), tr.next_power_of_2(channels))
    hash_kernel[(tr.cdiv(channels, 32),)](
        x,
        mean,
        planes,
        labels,
        tokens,
        channels,
        planes.shape[1],
        *x.stride(),
        planes.shape[0],
        32,
        ternary,
    )
    return labels


def groups(labels: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    '''Sort hash codes and compact occupied buckets without reading a device scalar.'''
    channels = labels.numel()
    sorted_codes, order = torch.sort(labels, stable=True)
    offsets = torch.empty(channels + 1, device=labels.device, dtype=torch.int32)
    count = torch.empty((), device=labels.device, dtype=torch.int32)
    compact_kernel[(1,)](sorted_codes, offsets, count, channels, tr.next_power_of_2(channels))
    return order, offsets, count


def partition(x: Tensor, planes: Tensor, ternary: bool = False) -> tuple[Tensor, Tensor, Tensor]:
    '''Return sorted channel indices, bucket offsets, and a device bucket count.'''
    return groups(hashes(x, planes, ternary))


def values(
    x: Tensor, order: Tensor, offsets: Tensor, count: Tensor, cap: int, window: int
) -> Tensor:
    '''Average original channel values over each occupied bucket.'''
    output = torch.empty((x.shape[0], cap), device=x.device, dtype=torch.float32)
    value_kernel[(cap,)](x, order, offsets, count, output, *x.shape, *x.stride(), cap, window)
    return output


def filters(
    weight: Tensor, order: Tensor, offsets: Tensor, count: Tensor, cap: int, base: int, rows: int
) -> Tensor:
    '''Sum weight columns over each occupied bucket for one output tile.'''
    output = torch.empty((rows, cap), device=weight.device, dtype=torch.float32)
    weight_kernel[(cap, tr.cdiv(rows, 32))](
        weight,
        order,
        offsets,
        count,
        output,
        weight.shape[1],
        weight.shape[0],
        base,
        rows,
        *weight.stride(),
        cap,
    )
    return output


def product(
    value: Tensor, weight: Tensor, count: Tensor, bias: Tensor | None, output: Tensor, base: int
) -> None:
    '''Write a reduced product tile with one bias addition per output element.'''
    product_kernel[(tr.cdiv(output.shape[0], 32), tr.cdiv(weight.shape[0], 32))](
        value,
        weight,
        count,
        bias,
        output,
        output.shape[0],
        output.shape[1],
        base,
        weight.shape[0],
        value.shape[1],
        output.stride(0),
        bias is not None,
    )


def linear(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    planes: Tensor,
    tile: int = 128,
    stats: Tensor | None = None,
    ternary: bool = False,
) -> Tensor:
    '''Evaluate one window at a time with bounded merged weight workspace.

    No device scalar is read by the host. Each product iterates only through
    occupied buckets. Sorting and launch overhead are included in benchmarks.
    '''
    if x.ndim != 3 or x.shape[-1] != weight.shape[-1]:
        raise ValueError('Expected input [batch, tokens, channels] matching the linear weight')
    if not x.is_cuda or x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError('CUDA compression requires BF16 or FP32 input')
    if weight.device != x.device or planes.device != x.device:
        raise ValueError('Input, weight, and projections must share a CUDA device')
    if x.shape[-1] > 32768:
        raise ValueError('The CUDA compaction kernel supports at most 32768 input channels')
    output = x.new_empty((*x.shape[:2], weight.shape[0]))
    window = planes.shape[0]
    cap = min(x.shape[-1], 1 << planes.shape[1])
    for batch in range(x.shape[0]):
        for start in range(0, x.shape[1], window):
            part = x[batch, start : start + window]
            order, offsets, count = partition(part, planes, ternary)
            value = values(part, order, offsets, count, cap, window)
            if stats is not None:
                stats[0].add_(count)
                stats[1].add_(1)
                stats[2].add_(part.shape[-1])
            for base in range(0, weight.shape[0], tile):
                rows = min(tile, weight.shape[0] - base)
                filt = filters(weight, order, offsets, count, cap, base, rows)
                product(value, filt, count, bias, output[batch, start : start + window], base)
    return output
