# SPDX-License-Identifier: Apache-2.0
"""Bench the tiered hot-set PLE embedding against the bare upstream mmap one.

Run on an idle machine (close other GPU/memory-heavy apps; `sudo purge`
between passes for true cold numbers). Builds a synthetic affine-U32 n-gram
table so no model download is needed:

    .venv/bin/python benchmarks/bench_qwen4_ple_hotset.py \
        --table-gib 4 --hot-mib 512 --passes 6

Point it at a real checkpoint slice instead of synthesizing:

    .venv/bin/python benchmarks/bench_qwen4_ple_hotset.py \
        --model-path /path/to/qwen4-exp --prefix language_model.model.layers.1...

Reports cold and warm rows/s for upstream vs tier; the tier must not regress
the cold path and must win the skewed warm path.
"""

from __future__ import annotations

import argparse
import json
import struct
import time
from pathlib import Path

import mlx.core as mx
import numpy as np


def _write_affine_table(out_dir: Path, gib: float, dims: int, num_shards: int):
    prefix = "ple.ngram_embedding"
    row_bytes = dims // 2 + dims // 32 * 4
    total_rows = max(int(gib * 1024**3 / row_bytes), num_shards * 1024)
    per_shard = (total_rows + num_shards - 1) // num_shards
    rng = np.random.default_rng(1234)
    packed_cols = dims * 4 // 32
    scale_cols = dims // 32
    weight_map = {}
    written = 0
    for shard in range(num_shards):
        rows = min(per_shard, total_rows - written)
        if rows <= 0:
            break
        fname = f"ple_shard_{shard}.safetensors"
        tensors = {
            f"{prefix}.shard_{shard}.weight": (
                rng.integers(0, 2**32, (rows, packed_cols), dtype=np.uint32),
                "U32",
            ),
            f"{prefix}.shard_{shard}.scales": (
                (rng.random((rows, scale_cols), np.float32).view(np.uint32) >> 16).astype(np.uint16),
                "BF16",
            ),
            f"{prefix}.shard_{shard}.biases": (
                (rng.random((rows, scale_cols), np.float32).view(np.uint32) >> 16).astype(np.uint16),
                "BF16",
            ),
        }
        header = {}
        offset = 0
        for key, (arr, dtype) in tensors.items():
            header[key] = {
                "dtype": dtype,
                "shape": list(arr.shape),
                "data_offsets": [offset, offset + arr.nbytes],
            }
            offset += arr.nbytes
        header_bytes = json.dumps(header).encode()
        pad = (-len(header_bytes)) % 8
        with open(out_dir / fname, "wb") as f:
            f.write(struct.pack("<Q", len(header_bytes) + pad))
            f.write(header_bytes + b" " * pad)
            for arr, _ in tensors.values():
                f.write(arr.tobytes())
        written += rows
        for suffix in ("weight", "scales", "biases"):
            weight_map[f"{prefix}.shard_{shard}.{suffix}"] = fname
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    return prefix, written, dims


def _token_stream(total_rows, length, seed, hot_span=0, hot_fraction=0.8):
    """One shared token id stream, consumed by both arms. Real prompts reuse
    n-grams hard: `hot_span` rows absorb `hot_fraction` of touches, the rest
    is a cold Zipf tail. The flat full-vocab Zipf that fooled #3235 is the
    --hot-span 0 corner."""
    rng = np.random.default_rng(seed)
    if hot_span <= 0:
        raw = rng.zipf(1.02, (length,)).astype(np.int64)
        return np.clip(raw, 0, total_rows - 1).astype(np.int64)
    hot_span = min(hot_span, total_rows)
    mask = rng.random(length) < hot_fraction
    hot = rng.integers(0, hot_span, (length,), dtype=np.int64)
    tail = rng.zipf(1.02, (length,)).astype(np.int64)
    tail = np.clip(tail, 0, total_rows - 1).astype(np.int64)
    return np.where(mask, hot, tail).astype(np.int64)


def _dirty_pressure(gib: float, seed: int):
    """Allocate and dirty anonymous RAM and HOLD it for the whole run. Living
    anon pages force the kernel to reclaim memory-mapped file pages, so the
    upstream arm re-faults cold rows from SSD every call while the hot set,
    already resident, does not. This is the regime the tier earns its keep in."""
    if gib <= 0:
        return None
    buf = bytearray(int(gib * 1024**3))
    step = max(len(buf) // 200_000, 1)
    buf[::step] = b"\x01" * ((len(buf) - 1) // step + 1)
    return buf


def _bench_stream(label, embed, stream, rows_per_call, pressure, seed=5):
    times = []
    hits = []
    for start in range(0, len(stream) - rows_per_call + 1, rows_per_call):
        ids = mx.array(stream[start : start + rows_per_call], dtype=mx.int64)
        t0 = time.perf_counter()
        out = embed(ids)
        mx.eval(out)
        times.append(time.perf_counter() - t0)
        if hasattr(embed, "hot"):
            hits.append(embed.hot.resident)
    times = np.asarray(times)
    half = max(len(times) // 2, 1)
    early = times[:half].mean()
    steady = times[half:].mean() if len(times) > half else early
    rps = rows_per_call / steady
    print(
        f"{label:22s} calls={len(times):4d} | early {early*1e3:7.2f} ms | "
        f"steady {steady*1e3:7.2f} ms ({rps:9.0f} rows/s)"
    )
    if hits:
        print(f"{'':22s} hot residency: start {hits[0]} -> end {hits[-1]}")
    del pressure
    return times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-gib", type=float, default=4.0)
    ap.add_argument("--hot-mib", type=float, default=512.0)
    ap.add_argument("--dims", type=int, default=512)
    ap.add_argument("--num-shards", type=int, default=2)
    ap.add_argument("--rows-per-call", type=int, default=64)
    ap.add_argument("--calls", type=int, default=512)
    ap.add_argument(
        "--pressure-gib",
        type=float,
        default=0.0,
        help="dirty+hold this much anonymous RAM for the whole run to force"
        " page-cache pressure (the regime where the hot set earns its keep)",
    )
    ap.add_argument(
        "--arm",
        choices=["both", "upstream", "tier"],
        default="both",
        help="run one arm per process; 'sudo purge' between arms for clean"
        " cold numbers (per #3235 protocol)",
    )
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument(
        "--hot-span",
        type=int,
        default=0,
        help="rows that absorb --hot-fraction of touches (0 = flat Zipf)",
    )
    ap.add_argument("--hot-fraction", type=float, default=0.8)
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--prefix", default=None)
    args = ap.parse_args()

    import tempfile

    from omlx.patches.mlx_vlm_qwen4_exp_compat import (
        apply_mlx_vlm_qwen4_exp_compat_patch,
    )
    from omlx.patches.qwen4_exp import DiskBackedShardedEmbedding as Tier

    tmp = Path(tempfile.mkdtemp(prefix="omlx-ple-bench-"))
    if args.model_path:
        # read geometry straight from the real index
        index = json.loads(
            (Path(args.model_path) / "model.safetensors.index.json").read_text()
        )
        prefix = args.prefix
        weights = [k for k in index["weight_map"] if k.startswith(prefix)]
        dims = args.dims
        total_rows = 0
        for w in weights:
            if w.endswith(".weight"):
                with open(Path(args.model_path) / index["weight_map"][w], "rb") as f:
                    (hlen,) = struct.unpack("<Q", f.read(8))
                    hdr = json.loads(f.read(hlen).decode().rstrip())
                    total_rows += hdr[w]["shape"][0]
        num_shards = sum(1 for w in weights if w.endswith(".weight"))
    else:
        prefix, total_rows, dims = _write_affine_table(
            tmp, args.table_gib, args.dims, args.num_shards
        )
        num_shards = args.num_shards
        print(f"synthetic table: {total_rows} rows x {dims} (~{args.table_gib} GiB)")

    apply_mlx_vlm_qwen4_exp_compat_patch()
    import mlx_vlm.models.qwen4_exp.language as language

    stream = _token_stream(
        total_rows,
        args.calls * args.rows_per_call,
        args.seed,
        args.hot_span,
        args.hot_fraction,
    )
    common = (
        tmp if not args.model_path else Path(args.model_path),
        prefix,
        total_rows,
        dims,
        num_shards,
    )
    print(f"access: {args.calls} calls x {args.rows_per_call} rows, "
          f"pressure {args.pressure_gib} GiB, arm={args.arm}")
    if args.arm == "both":
        print("NOTE: one process warms the cache for the later arm; for clean"
              " cold numbers run --arm upstream, 'sudo purge', then --arm tier.")
    pressure = _dirty_pressure(args.pressure_gib, args.seed)

    if args.arm in ("both", "upstream"):
        bare = language.DiskBackedShardedEmbedding(*common)
        _bench_stream(
            "upstream", bare, stream, args.rows_per_call, pressure, args.seed
        )
        bare.close()
        del bare

    if args.arm in ("both", "tier"):
        hot_bytes = int(args.hot_mib * 1024**2)
        tier = Tier(*common, hot_set_bytes=hot_bytes)
        print(f"hot set budget: {tier._max_entries} rows ({hot_bytes / 1024**2:.0f} MiB)")
        _bench_stream(
            "tier", tier, stream, args.rows_per_call, pressure, args.seed
        )
        tier.close()
        del tier


if __name__ == "__main__":
    main()
