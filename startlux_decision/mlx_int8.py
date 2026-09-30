"""int8 matmuls for MLXDecision(int8=True) on the neural accelerators of M5 and later GPUs.

The large projections (MLP, attention and linear-attention in/out projections) run as int8 x int8 -> int32 through Metal
4's MetalPerformancePrimitives matmul2d on the M5's neural accelerators.  Activations are quantized per token on every
call, weights per output channel once at load.  SmoothQuant (alpha 0.75) first moves
activation outliers into the weights, using activation ranges from the built-in CALIBRATION requests, which contain no
benchmark items.  The embedding, the output head and projections too small for the kernel stay as they are.

The quantize and matmul kernels are adapted from JetBrains/mlx-vlm (branch feature/int8-prefill, mlx_vlm/int8_prefill.py),
released under the MIT License:

    Copyright © 2025 Prince Canuma

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
    persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
    Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
    WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
    COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
    OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import re

import mlx.core as mx
import mlx.nn as nn

ALPHA = 0.75                # SmoothQuant migration strength
TILE = (64, 128, 4)         # output tile rows, columns and simdgroups; requests are tens to hundreds of tokens
MIN_DIM = 1024              # projections smaller than this in either dimension stay as they are

_HEADER = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
using namespace metal;
"""

# One threadgroup per row: absmax of x * inv_s over the row, then symmetric int8 with that row's scale.
_QUANT_SRC = """
    constexpr int K = {K};
    constexpr int NTH = 256;
    uint row = threadgroup_position_in_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = tid % 32;
    uint sg = tid / 32;
    const device {T}* xrow = x + size_t(row) * K;
    float amax = 0.0f;
    for (int i = tid; i < K; i += NTH) amax = max(amax, fabs(float(xrow[i]) * inv_s[i]));
    amax = simd_max(amax);
    threadgroup float tg_max[NTH / 32];
    if (lane == 0) tg_max[sg] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    amax = tg_max[lane % (NTH / 32)];
    amax = simd_max(amax);
    float scale = max(amax, 1e-8f) / 127.0f;
    float inv = 1.0f / scale;
    if (tid == 0) xs[row] = scale;
    device int8_t* qrow = xq + size_t(row) * K;
    for (int i = tid; i < K; i += NTH) qrow[i] = int8_t(clamp(rint(float(xrow[i]) * inv_s[i] * inv), -127.0f, 127.0f));
"""

# One threadgroup per TM x TN output tile; matmul2d loops over K.  Edge rows are bounds-checked, so any M works.
_GEMM_SRC = """
    constexpr int N = {N};
    constexpr int K = {K};
    constexpr int TM = {TM};
    constexpr int TN = {TN};
    uint2 tgid = threadgroup_position_in_grid.xy;
    const int M = m_dim[0];
    constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent), false, true, true,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<{NS}>> op;
    auto A = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>((device int8_t*)xq, dextents<int32_t, 2>(K, M));
    auto B = tensor<device int8_t, dextents<int32_t, 2>, tensor_inline>((device int8_t*)wq, dextents<int32_t, 2>(K, N));
    auto tA = A.slice(0, int(tgid.y) * TM);
    auto tB = B.slice(0, int(tgid.x) * TN);
    auto cT = op.get_destination_cooperative_tensor<decltype(tA), decltype(tB), int32_t>();
#pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0;
    op.run(tA, tB, cT);
#pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {{
        if (cT.is_valid_element(i)) {{
            auto idx = cT.get_multidimensional_index(i);
            int n = int(tgid.x) * TN + idx[0];
            int m = int(tgid.y) * TM + idx[1];
            if (m < M && n < N) out[size_t(m) * N + n] = bfloat(float(cT[i]) * xs[m] * ws[n]);
        }}
    }}
"""

_kernels = {}


def supported():
    """True on GPUs with neural accelerators (Apple GPU family 17, the M5, and later)."""
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    m = re.match(r"applegpu_g(\d+)", info.get("architecture", ""))
    return bool(m) and int(m.group(1)) >= 17


def _quantize_rows(x, inv_s):
    M, K = x.shape
    key = ("q", K)
    if key not in _kernels:
        _kernels[key] = mx.fast.metal_kernel(name=f"sld_quant_{K}", input_names=["x", "inv_s"], output_names=["xq", "xs"],
                                             source=_QUANT_SRC.format(K=K, T="bfloat"), header=_HEADER)
    return _kernels[key](inputs=[x, inv_s], output_shapes=[(M, K), (M,)], output_dtypes=[mx.int8, mx.float32],
                         grid=(M * 256, 1, 1), threadgroup=(256, 1, 1))


def _matmul(xq, xs, wq, ws):
    TM, TN, NS = TILE
    M, K = xq.shape
    N = wq.shape[0]
    key = ("g", N, K)
    if key not in _kernels:
        _kernels[key] = mx.fast.metal_kernel(name=f"sld_i8mm_{N}_{K}", input_names=["xq", "xs", "wq", "ws", "m_dim"],
                                             output_names=["out"], header=_HEADER,
                                             source=_GEMM_SRC.format(N=N, K=K, TM=TM, TN=TN, NS=NS))
    out, = _kernels[key](inputs=[xq, xs, wq, ws, mx.array([M], dtype=mx.int32)], output_shapes=[(M, N)],
                         output_dtypes=[mx.bfloat16], grid=((N // TN) * 32 * NS, (M + TM - 1) // TM, 1),
                         threadgroup=(32 * NS, 1, 1))
    return out


class Int8Linear(nn.Module):
    """nn.Linear (or an mlx-lm QuantizedLinear) as a W8A8 matmul with SmoothQuant scales."""

    def __init__(self, linear, act_max):
        super().__init__()
        w = linear.weight
        if hasattr(linear, "scales"):
            w = mx.dequantize(w, linear.scales, linear.biases, linear.group_size, linear.bits)
        w = w.astype(mx.float32)
        s = mx.clip(mx.power(mx.maximum(act_max, 1e-5), ALPHA) / mx.power(mx.maximum(mx.abs(w).max(axis=0), 1e-5), 1 - ALPHA),
                    1e-3, 1e3)
        w = w * s[None, :]
        self.inv_s = 1.0 / s
        self.ws = mx.maximum(mx.abs(w).max(axis=1), 1e-8) / 127.0
        self.wq = mx.clip(mx.round(w / self.ws[:, None]), -127, 127).astype(mx.int8)
        self.bias = linear.bias if "bias" in linear else None
        self.freeze()

    def __call__(self, x):
        shape = x.shape
        x2 = x.reshape(-1, shape[-1]).astype(mx.bfloat16)
        y = _matmul(*_quantize_rows(x2, self.inv_s), self.wq, self.ws)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*shape[:-1], self.wq.shape[0])


def _eligible(layer):
    n, k = layer.weight.shape
    k = k * 32 // layer.bits if hasattr(layer, "scales") else k
    return n % TILE[1] == 0 and k % 32 == 0 and min(n, k) >= MIN_DIM


def _projections(body):
    return [(name, mod, cname, child) for name, mod in body.named_modules() if not isinstance(mod, _Recorder)
            for cname, child in mod.children().items()
            if isinstance(child, (nn.Linear, nn.QuantizedLinear)) and _eligible(child)]


class _Recorder(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner, self.amax = inner, None

    def __call__(self, x):
        m = mx.abs(x.reshape(-1, x.shape[-1]).astype(mx.float32)).max(axis=0)
        self.amax = m if self.amax is None else mx.maximum(self.amax, m)
        return self.inner(x)


def convert(body, sequences):
    """Replace the eligible projections of `body` by Int8Linear, calibrated on `sequences` (token id lists); returns
    how many were replaced.  Layers are converted one at a time, so a lazily loaded model never holds two copies."""
    slots = [(mod, cname, _Recorder(child)) for _, mod, cname, child in _projections(body)]
    for mod, cname, rec in slots:
        setattr(mod, cname, rec)
    for ids in sequences:
        body(mx.array([ids]))
        mx.eval([rec.amax for _, _, rec in slots])
    for i, (mod, cname, rec) in enumerate(slots):
        new = Int8Linear(rec.inner, rec.amax)
        mx.eval(new.parameters())
        setattr(mod, cname, new)
        if i % 16 == 15:
            mx.clear_cache()
    mx.clear_cache()
    return len(slots)


CALIBRATION = []            # (state, questions) pairs of the kinds of requests the models serve
for _t in ("My parcel was marked delivered but never arrived.", "The app crashes when I open settings on Android 15.",
           "I want a refund for the duplicate charge on invoice 7781.", "Can I change the shipping address after ordering?",
           "Password reset emails never arrive.", "The discount code SPRING20 is rejected at checkout."):
    CALIBRATION.append(({"ticket": _t}, {
        "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
                 "criteria": {"billing": "Payments, refunds and invoices", "shipping": "Delivery and tracking",
                              "technical": "App, login and account problems"}},
        "urgent": {"type": "noul", "instructions": "Should this ticket be answered today?"},
        "severity": {"type": "score", "instructions": "How severe is the impact?",
                     "criteria": ["cosmetic", "annoying", "blocks the customer"]}}))
for _n in (5, 20, 60):
    CALIBRATION.append(({"page": " ".join(f"Item {i}: {['laptop', 'desk lamp', 'usb cable', 'office chair'][i % 4]} "
                                          f"${(i * 37) % 900}.99, {'in stock' if i % 3 else 'backorder'}." for i in range(_n)),
                         "goal": "Buy the cheapest in-stock chair."},
                        {"next": {"type": "choice", "instructions": "Which item should be clicked next?",
                                  "criteria": {f"item{i}": f"Item {i}" for i in range(min(_n, 12))}},
                         "done": {"type": "noul", "instructions": "Is the goal already met?"}}))
for _r in ("The hotel was spotless and the staff helpful.", "Food arrived cold and two items were missing.",
           "Average experience, nothing special.", "Absolutely terrible support, never again!"):
    CALIBRATION.append(({"review": _r}, {
        "stars": {"type": "score", "instructions": "How satisfied is the customer?", "criteria": ["1", "2", "3", "4", "5"]},
        "topic": {"type": "choice", "instructions": "What is the review mainly about?",
                  "criteria": {"service": "staff and support", "product": "food or room", "price": "cost",
                               "delivery": "delivery"}}}))
for _q, _o in (("Which planet is known as the Red Planet?", ["Venus", "Mars", "Jupiter", "Saturn"]),
               ("What is 17 * 3?", ["41", "51", "57", "61"]), ("Who wrote Hamlet?", ["Dickens", "Shakespeare", "Austen", "Twain"])):
    CALIBRATION.append(({"quiz": _q}, {"answer": {"type": "choice", "instructions": _q, "criteria": {o.lower(): o for o in _o}}}))
CALIBRATION.append(({"log": "\n".join(f"2026-09-30 12:{i:02d} {'ERROR' if i % 7 == 0 else 'INFO'} worker-{i % 3} step {i}"
                                      for i in range(40))},
                    {"failing": {"type": "noul", "instructions": "Is any worker failing repeatedly?"},
                     "worker": {"type": "choice", "instructions": "Which worker logs the most errors?",
                                "criteria": {"w0": "worker-0", "w1": "worker-1", "w2": "worker-2"}}}))
