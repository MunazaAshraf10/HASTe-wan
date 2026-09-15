import triton
import triton.language as tl


@triton.jit
def assign(
    X,
    CENTERS,
    LABELS,
    R,
    N: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    T: tl.constexpr,
    C: tl.constexpr,
):
    h = tl.program_id(0)
    if tl.load(R + h):
        t = tl.program_id(1) * T + tl.arange(0, T)
        d = tl.arange(0, C)
        x = tl.load(
            X + (h * N + t[:, None]) * D + d[None, :], (t[:, None] < N) & (d[None, :] < D), 0
        ).to(tl.float32)
        best = tl.full((T,), float('inf'), tl.float32)
        label = tl.full((T,), 0, tl.int32)
        norm = tl.sum(x * x, 1)
        for start in range(0, K, 32):
            indices = start + tl.arange(0, 32)
            centers = tl.load(
                CENTERS + (h * K + indices[None, :]) * D + d[:, None],
                (indices[None, :] < K) & (d[:, None] < D),
                0,
            )
            distance = (
                norm[:, None]
                + tl.sum(centers * centers, 0)[None, :]
                - 2 * tl.dot(x, centers, input_precision='tf32x3')
            )
            distance = tl.where(indices[None, :] < K, distance, float('inf'))
            value = tl.min(distance, 1)
            index = tl.min(tl.where(distance == value[:, None], indices[None, :], K), 1)
            take = value < best
            label = tl.where(take, index, label)
            best = tl.minimum(best, value)
        tl.store(LABELS + h * N + t, label, t < N)


@triton.jit
def centroid(
    X,
    ORDER,
    OFFSETS,
    OLD,
    OUT,
    R,
    N: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    T: tl.constexpr,
    C: tl.constexpr,
):
    h = tl.program_id(0)
    c = tl.program_id(1)
    if tl.load(R + h):
        d = tl.arange(0, C)
        total = tl.full((C,), 0, tl.float32)
        count = tl.full((), 0, tl.float32)
        begin = tl.load(OFFSETS + h * (K + 1) + c)
        end = tl.load(OFFSETS + h * (K + 1) + c + 1)
        for start in range(begin, end, T):
            pos = start + tl.arange(0, T)
            t = tl.load(ORDER + h * N + pos, pos < end, 0)
            x = tl.load(
                X + (h * N + t[:, None]) * D + d[None, :],
                (pos[:, None] < end) & (d[None, :] < D),
                0,
            ).to(tl.float32)
            total += tl.sum(x, 0)
        count = (end - begin).to(tl.float32)
        old = tl.load(OLD + (h * K + c) * D + d, d < D, 0)
        tl.store(
            OUT + (h * K + c) * D + d, tl.where(count > 0, total / tl.maximum(count, 1), old), d < D
        )


@triton.jit
def histogram(
    L,
    HIST,
    R,
    N: tl.constexpr,
    K: tl.constexpr,
    BINS: tl.constexpr,
    NQ: tl.constexpr,
    AREA: tl.constexpr,
    RA: tl.constexpr,
    QUERY: tl.constexpr,
    T: tl.constexpr,
):
    h = tl.program_id(0)
    c = tl.program_id(1)
    f = tl.program_id(2)
    if tl.load(R + h):
        count = 0
        for start in range(0, N, T):
            t = start + tl.arange(0, T)
            label = tl.load(L + h * N + t, t < N, -1)
            if QUERY:
                frame = t // AREA
            else:
                frame = tl.where(t < NQ, BINS - 1, (t - NQ) // RA + 1)
            count += tl.sum(((t < N) & (label == c) & (frame == f)).to(tl.int32), 0)
        tl.store(HIST + (h * K + c) * BINS + f, count)


@triton.jit
def semantic(
    Q,
    K,
    QH,
    KH,
    SCORE,
    PAIRS,
    GEN,
    R,
    QC: tl.constexpr,
    KC: tl.constexpr,
    D: tl.constexpr,
    BINS: tl.constexpr,
    FRAMES: tl.constexpr,
    C: tl.constexpr,
    TILE: tl.constexpr,
):
    h = tl.program_id(0)
    a = tl.program_id(1)
    b = tl.program_id(2) * TILE + tl.arange(0, TILE)
    if tl.load(R + h):
        d = tl.arange(0, C)
        q = tl.load(Q + (h * QC + a) * D + d, d < D, 0)
        k = tl.load(
            K + (h * KC + b[:, None]) * D + d[None, :], (b[:, None] < KC) & (d[None, :] < D), 0
        )
        logits = tl.sum(q[None, :] * k, 1) * (D**-0.5)
        generation = tl.load(KH + (h * KC + b) * BINS + BINS - 1, b < KC, 0)
        qsize = 0.0
        permitted = tl.full((TILE,), 0, tl.float32)
        for f in range(FRAMES):
            qcount = tl.load(QH + (h * QC + a) * BINS + f)
            kcount = tl.load(KH + (h * KC + b) * BINS + f, b < KC, 0)
            qsize += qcount
            permitted += qcount * kcount
        permitted += qsize * generation
        logits += tl.log(tl.maximum(permitted / tl.maximum(qsize, 1), 1e-30))
        tl.store(
            SCORE + (h * QC + a) * KC + b, tl.where(permitted > 0, logits, -float('inf')), b < KC
        )
        tl.store(PAIRS + (h * QC + a) * KC + b, permitted, b < KC)
        if a == 0:
            tl.store(GEN + h * KC + b, generation, b < KC)


@triton.jit
def xestimate(
    Q,
    K,
    OUT,
    R,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    D: tl.constexpr,
    AREA: tl.constexpr,
    RA: tl.constexpr,
    BLOCK: tl.constexpr,
    STRIDE: tl.constexpr,
    START: tl.constexpr,
    ROWS: tl.constexpr,
    C: tl.constexpr,
    TILE: tl.constexpr,
):
    h = tl.program_id(0)
    local = tl.program_id(1)
    g = START + local
    b = tl.program_id(2) * TILE + tl.arange(0, TILE)
    if tl.load(R + h):
        d = tl.arange(0, C)
        score = tl.full((TILE,), 0, tl.float32)
        count = tl.full((TILE,), 0, tl.float32)
        for s in range(STRIDE):
            qi = g * STRIDE + STRIDE - 1 - s
            ki = b * STRIDE + s
            valid = (qi < NQ) & (ki < NK) & ((ki < NQ) | (qi // AREA == (ki - NQ) // RA + 1))
            q = tl.load(Q + (h * NQ + qi) * D + d, (qi < NQ) & (d < D), 0).to(tl.float32)
            k = tl.load(
                K + (h * NK + ki[:, None]) * D + d[None, :],
                (ki[:, None] < NK) & (d[None, :] < D),
                0,
            ).to(tl.float32)
            score += tl.where(valid, tl.sum(q[None, :] * k, 1), 0)
            count += valid.to(tl.float32)
        logits = score / tl.maximum(count, 1) * (D**-0.5)
        tl.store(
            OUT + (h * ROWS + local) * tl.cdiv(NK, STRIDE) + b,
            tl.where(count > 0, logits, -float('inf')),
            b < tl.cdiv(NK, STRIDE),
        )


@triton.jit
def normalize(X, OUT, R, ROWS: tl.constexpr, COLS: tl.constexpr, C: tl.constexpr):
    h = tl.program_id(0)
    row = tl.program_id(1)
    if tl.load(R + h):
        c = tl.arange(0, C)
        x = tl.load(X + (h * ROWS + row) * COLS + c, c < COLS, -float('inf'))
        peak = tl.max(x, 0)
        p = tl.exp(x - tl.where(peak == -float('inf'), 0, peak))
        p = p / tl.maximum(tl.sum(p, 0), 1e-30)
        tl.store(OUT + (h * ROWS + row) * COLS + c, p, c < COLS)


@triton.jit
def xreduce(
    P,
    SCORE,
    R,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    BLOCK: tl.constexpr,
    STRIDE: tl.constexpr,
    START: tl.constexpr,
    ROWS: tl.constexpr,
    S: tl.constexpr,
):
    h = tl.program_id(0)
    a = tl.program_id(1)
    b = tl.program_id(2)
    if tl.load(R + h):
        qi = a * (BLOCK // STRIDE) + tl.arange(0, S)
        ki = b * (BLOCK // STRIDE) + tl.arange(0, S)
        p = tl.load(
            P + (h * ROWS + qi[:, None]) * tl.cdiv(NK, STRIDE) + ki[None, :],
            (qi[:, None] < ROWS) & (ki[None, :] < tl.cdiv(NK, STRIDE)),
            0,
        )
        tl.store(
            SCORE + (h * tl.cdiv(NQ, BLOCK) + a + START) * tl.cdiv(NK, BLOCK) + b,
            tl.sum(tl.sum(p, 0), 0),
        )


@triton.jit
def choose(
    S, MASK, GEN, P, R, QC: tl.constexpr, KC: tl.constexpr, MINIMUM: tl.constexpr, C: tl.constexpr
):
    h = tl.program_id(0)
    row = tl.program_id(1)
    if tl.load(R + h):
        i = tl.arange(0, C)
        scores = tl.load(S + (h * QC + row) * KC + i, i < KC, 0)
        # Stable ties use the original cluster index.
        key = (scores.to(tl.uint32, bitcast=True).to(tl.uint64) << 32) + (C - i).to(tl.uint64)
        sorted_key, order = tl.sort_pairs(key, i, descending=True)
        ranked = tl.load(S + (h * QC + row) * KC + order, order < KC, 0)
        total = tl.sum(ranked, 0)
        before = tl.cumsum(ranked, 0) - ranked
        keep = (before < total * tl.load(P + h)) & (ranked > 0)
        keep |= (i < int(KC * MINIMUM)) & (ranked > 0)
        keep |= (tl.load(P + h) >= 1) & (ranked > 0)
        gen = tl.load(GEN + h * KC + order, order < KC, 0) > 0
        anygen = tl.sum((keep & gen).to(tl.int32), 0) > 0
        firstgen = tl.min(tl.where(gen, i, C), 0)
        keep |= (~anygen) & (i == firstgen)
        tl.store(MASK + (h * QC + row) * KC + order, keep, order < KC)


@triton.jit
def attend(
    Q,
    K,
    V,
    QORDER,
    KORDER,
    QLABEL,
    QOFF,
    KOFF,
    MASK,
    OUT,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    D: tl.constexpr,
    QC: tl.constexpr,
    KC: tl.constexpr,
    AREA: tl.constexpr,
    RA: tl.constexpr,
    C: tl.constexpr,
    T: tl.constexpr,
    U: tl.constexpr,
):
    h = tl.program_id(0)
    tile = tl.program_id(1)
    # Sorted query tiles may straddle clusters; test the selected mask per row.
    pos = tile * T + tl.arange(0, T)
    qi = tl.load(QORDER + h * NQ + pos, pos < NQ, 0)
    label = tl.load(QLABEL + h * NQ + qi, pos < NQ, 0)
    d = tl.arange(0, C)
    q = tl.load(
        Q + (h * NQ + qi[:, None]) * D + d[None, :], (pos[:, None] < NQ) & (d[None, :] < D), 0
    )
    peak = tl.full((T,), -float('inf'), tl.float32)
    denom = tl.full((T,), 0, tl.float32)
    accum = tl.full((T, C), 0, tl.float32)
    for cluster in range(KC):
        selected = tl.load(MASK + (h * QC + label) * KC + cluster) & (pos < NQ)
        if tl.sum(selected.to(tl.int32), 0) > 0:
            start = tl.load(KOFF + h * (KC + 1) + cluster)
            end = tl.load(KOFF + h * (KC + 1) + cluster + 1)
            for offset in range(start, end, U):
                kp = offset + tl.arange(0, U)
                ki = tl.load(KORDER + h * NK + kp, kp < end, 0)
                k = tl.load(
                    K + (h * NK + ki[None, :]) * D + d[:, None],
                    (kp[None, :] < end) & (d[:, None] < D),
                    0,
                )
                v = tl.load(
                    V + (h * NK + ki[:, None]) * D + d[None, :],
                    (kp[:, None] < end) & (d[None, :] < D),
                    0,
                )
                valid = (
                    selected[:, None]
                    & (kp[None, :] < end)
                    & ((ki[None, :] < NQ) | (qi[:, None] // AREA == (ki[None, :] - NQ) // RA + 1))
                )
                logits = tl.dot(q, k).to(tl.float32) * (D**-0.5)
                logits = tl.where(valid, logits, -float('inf'))
                updated = tl.maximum(peak, tl.max(logits, 1))
                safe = tl.where(updated == -float('inf'), 0, updated)
                alpha = tl.exp(peak - safe)
                prob = tl.exp(logits - safe[:, None])
                accum = accum * alpha[:, None] + tl.dot(prob.to(v.dtype), v)
                denom = denom * alpha + tl.sum(prob, 1)
                peak = updated
    output = accum / tl.maximum(denom[:, None], 1e-30)
    tl.store(
        OUT + (h * NQ + qi[:, None]) * D + d[None, :],
        output,
        (pos[:, None] < NQ) & (d[None, :] < D),
    )


@triton.jit
def sizes(L, SIZES, R, N: tl.constexpr, K: tl.constexpr, T: tl.constexpr):
    h = tl.program_id(0)
    c = tl.program_id(1)
    if tl.load(R + h):
        count = 0
        for start in range(0, N, T):
            t = start + tl.arange(0, T)
            label = tl.load(L + h * N + t, t < N, -1)
            count += tl.sum(((t < N) & (label == c)).to(tl.int32), 0)
        tl.store(SIZES + h * K + c, count)


@triton.jit
def offsets(SIZES, OFFSETS, R, K: tl.constexpr, C: tl.constexpr):
    h = tl.program_id(0)
    if tl.load(R + h):
        c = tl.arange(0, C)
        count = tl.load(SIZES + h * K + c, c < K, 0)
        before = tl.cumsum(count, 0) - count
        tl.store(OFFSETS + h * (K + 1) + c, before, c < K)
        tl.store(OFFSETS + h * (K + 1) + K, tl.sum(count, 0))


@triton.jit
def scatter(L, ORDER, OFFSETS, R, N: tl.constexpr, K: tl.constexpr, T: tl.constexpr):
    h = tl.program_id(0)
    c = tl.program_id(1)
    if tl.load(R + h):
        cursor = tl.load(OFFSETS + h * (K + 1) + c)
        for start in range(0, N, T):
            t = start + tl.arange(0, T)
            label = tl.load(L + h * N + t, t < N, -1)
            use = (t < N) & (label == c)
            rank = tl.cumsum(use.to(tl.int32), 0) - 1
            tl.store(ORDER + h * N + cursor + rank, t, use)
            cursor += tl.sum(use.to(tl.int32), 0)
