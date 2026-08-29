# SPDX-License-Identifier: Apache-2.0
"""Tests for the tiered hot-set overlay on the qwen4_exp PLE table."""

from __future__ import annotations

import json
import struct

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.qwen4_exp import DiskBackedShardedEmbedding, NGramSlotCache

DIMS = 160
SHARD_ROWS = 64
PREFIX = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding"


def _bf16_bits(values: np.ndarray) -> np.ndarray:
    return (values.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def _write_safetensors(path, tensors: dict[str, tuple[np.ndarray, str]]):
    offset = 0
    header = {}
    for key, (array, dtype) in tensors.items():
        header[key] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [offset, offset + array.nbytes],
        }
        offset += array.nbytes
    header_bytes = json.dumps(header).encode()
    pad = (-len(header_bytes)) % 8
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes) + pad))
        f.write(header_bytes + b" " * pad)
        for array, _dtype in tensors.values():
            f.write(array.tobytes())


def _write_affine_shard(path, num_rows: int, keybase: str, rng, dims: int = DIMS):
    packed_cols = dims * 4 // 32  # 4-bit packing
    scale_cols = dims // 32  # group size 32
    packed = rng.integers(0, 2**32, (num_rows, packed_cols), dtype=np.uint32)
    scales = _bf16_bits(rng.normal(1.0, 0.1, (num_rows, scale_cols)))
    biases = _bf16_bits(rng.normal(0.0, 0.05, (num_rows, scale_cols)))
    _write_safetensors(
        path,
        {
            f"{keybase}.weight": (packed, "U32"),
            f"{keybase}.scales": (scales, "BF16"),
            f"{keybase}.biases": (biases, "BF16"),
        },
    )
    return {"dtype": "U32", "weight": packed, "scales": scales, "biases": biases}


def _write_dense_shard(path, num_rows: int, keybase: str, rng, dims: int = DIMS):
    bits = _bf16_bits(rng.normal(0.0, 1.0, (num_rows, dims)))
    _write_safetensors(path, {f"{keybase}.weight": (bits, "BF16")})
    return {"dtype": "BF16", "weight": bits}


def _cat_shards(shards):
    merged = {}
    for field in ("weight", "scales", "biases"):
        parts = [s[field] for s in shards if field in s]
        if parts:
            merged[field] = np.concatenate(parts, axis=0)
    merged["dtype"] = shards[0]["dtype"]
    return merged


def _expected_rows(tensor, indices):
    if tensor["dtype"] == "U32":
        s = mx.array(
            (tensor["scales"].astype(np.uint32) << 16).view(np.float32)
        ).astype(mx.bfloat16)
        b = mx.array(
            (tensor["biases"].astype(np.uint32) << 16).view(np.float32)
        ).astype(mx.bfloat16)
        full = mx.dequantize(
            mx.array(tensor["weight"]),
            s,
            b,
            group_size=32,
            bits=4,
            mode="affine",
        )
    else:
        full = mx.array(
            (tensor["weight"].astype(np.uint32) << 16).view(np.float32)
        ).astype(mx.bfloat16)
    return mx.stack([full[int(i)] for i in indices], axis=0)


def _tier_tensors(tmp_path, keybases, writer):
    rng = np.random.default_rng(7)
    shards = []
    weight_map = {}
    for i, keybase in enumerate(keybases):
        fname = f"shard{i}.safetensors"
        shard = writer(tmp_path / fname, SHARD_ROWS, keybase, rng)
        shards.append(shard)
        for suffix in ("weight", "scales", "biases"):
            if suffix in shard:
                weight_map[f"{keybase}.{suffix}"] = fname
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    return _cat_shards(shards)


@pytest.fixture
def affine_tensors(tmp_path):
    keybases = [f"{PREFIX}.shard_{i}" for i in range(2)]
    return tmp_path, _tier_tensors(tmp_path, keybases, _write_affine_shard)


@pytest.fixture
def dense_tensors(tmp_path):
    keybases = [f"{PREFIX}.shard_{i}" for i in range(2)]
    return tmp_path, _tier_tensors(tmp_path, keybases, _write_dense_shard)


def _embed(tmp_path, tensors, **kwargs):
    return DiskBackedShardedEmbedding(
        tmp_path, PREFIX, 2 * SHARD_ROWS, DIMS, 2, **kwargs
    )


@pytest.fixture(scope="module", autouse=True)
def _tiny_hot_set(monkeypatch_session):
    from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    # Keep the preallocated hot table tiny in tests (default is 2 GiB).
    monkeypatch_session.setattr(
        DiskBackedShardedEmbedding, "_HOT_SET_BYTES", 1 << 20
    )


@pytest.fixture(scope="session")
def monkeypatch_session():
    # module-scoped plain swap: record and restore manually
    saved = []

    class _Swap:
        def setattr(self, obj, name, value):
            saved.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)

        def restore(self):
            for obj, name, value in reversed(saved):
                setattr(obj, name, value)

    swap = _Swap()
    yield swap
    swap.restore()


def test_call_populates_hot_set_and_hits(affine_tensors):
    tmp_path, tensors = affine_tensors
    emb = _embed(tmp_path, tensors)
    mid = SHARD_ROWS
    idx = mx.array([[0, 3, mid + 2], [7, mid, 1]], dtype=mx.int64)
    out1 = emb(idx)
    assert out1.shape == (2, 3, DIMS)
    assert emb.rows_read == 6  # every row missed the hot set

    expected = _expected_rows(tensors, [0, 3, mid + 2, 7, mid, 1])
    assert mx.array_equal(out1.reshape(6, DIMS), expected).item()

    out2 = emb(idx)
    assert emb.rows_read == 6  # second call fully served from the hot set
    assert mx.array_equal(out1, out2).item()
    assert emb.lru_cache.size == 6


def test_bf16_dense_checkpoint_serves_correct_rows(dense_tensors):
    """The tier must serve whatever dtype the checkpoint stores, not just
    affine-packed U32 (maintainer repro: BF16 crashed the first lookup)."""
    tmp_path, tensors = dense_tensors
    emb = _embed(tmp_path, tensors)
    mid = SHARD_ROWS
    ids = [0, 5, mid, mid + 7, 2 * SHARD_ROWS - 1]
    out = emb(mx.array([ids], dtype=mx.int64)).reshape(len(ids), DIMS)
    assert out.shape == (len(ids), DIMS)
    assert mx.array_equal(out, _expected_rows(tensors, ids)).item()

    out2 = emb(mx.array([ids], dtype=mx.int64)).reshape(len(ids), DIMS)
    assert mx.array_equal(out, out2).item()  # hot set holds the same rows


def test_hot_set_budget_bounds_the_whole_table(affine_tensors):
    tmp_path, _ = affine_tensors
    row_bytes = DIMS * 2
    emb = _embed(tmp_path, None, hot_set_bytes=4 * row_bytes)
    assert emb._max_entries == 4
    assert emb.hot.capacity == 4
    emb(mx.array([0, 1, 2, 3, 4, 5], dtype=mx.int64))
    # LRU + pinned never exceed the budget the table was sized from.
    assert emb.hot.resident == 4
    assert emb.hot.table.shape == (4, DIMS)


def test_slot_cache_resident_bounded_and_clear():
    """Slots must stay inside the single table — an out-of-table slot reads
    back as silent zeros on Metal — and clear() must restore every slot
    (pinned-region ones included) to the freelist."""
    hot = NGramSlotCache(capacity=4, dims=4)
    rows = mx.ones((4, 4), dtype=mx.bfloat16)
    hot.fill([0, 1, 2, 3], rows)
    assert hot.resident == 4 and len(hot._free) == 0

    hot.promote(0)
    hot.promote(1)
    assert hot.pinned_size == 2 and hot.lru_size == 2

    # Overflowing fills evict the coldest LRU rows but never overrun the
    # [capacity, dims] table nor touch the pins.
    other = mx.full((2, 4), 2.0, dtype=mx.bfloat16)
    hot.fill([4, 5], other)
    assert hot.resident == 4
    assert hot.pinned_size == 2
    assert all(0 <= slot < 4 for slot in hot._slot_of.values())

    hot.clear()
    assert sorted(hot._free) == list(range(4))
    assert hot.resident == 0


def test_missing_shard_fails_closed(tmp_path):
    """A shard missing from the index must fail loudly — zero-filled rows
    silently poison every downstream hidden state."""
    rng = np.random.default_rng(11)
    _write_affine_shard(
        tmp_path / "shard0.safetensors", SHARD_ROWS, f"{PREFIX}.shard_0", rng
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    f"{PREFIX}.shard_0.{key}": "shard0.safetensors"
                    for key in ("weight", "scales", "biases")
                }
            }
        )
    )
    with pytest.raises(KeyError, match="shard 1"):
        _embed(tmp_path, None)


def test_alt_key_style_shards_dot_index(tmp_path):
    rng = np.random.default_rng(3)
    n = 22
    prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    tensor = _write_affine_shard(
        tmp_path / "shard0.safetensors", n, f"{prefix}.shards.0", rng, dims=32
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    f"{prefix}.shards.0.{key}": "shard0.safetensors"
                    for key in ("weight", "scales", "biases")
                }
            }
        )
    )
    emb = DiskBackedShardedEmbedding(
        tmp_path, prefix, n, 32, 1
    )
    rows = emb.read_rows([0, 21])
    expected = _expected_rows(tensor, [0, 21])
    assert mx.array_equal(mx.stack(rows, axis=0), expected).item()


def test_shard_row_count_mismatch_raises(tmp_path):
    rng = np.random.default_rng(4)
    prefix = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding"
    _write_affine_shard(
        tmp_path / "s0.safetensors", 10, f"{prefix}.shard_0", rng, dims=32
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    f"{prefix}.shard_0.{key}": "s0.safetensors"
                    for key in ("weight", "scales", "biases")
                }
            }
        )
    )
    with pytest.raises(ValueError, match="Unexpected shape"):
        DiskBackedShardedEmbedding(tmp_path, prefix, 22, 32, 1)


def test_weight_scale_parameter_and_application(affine_tensors):
    tmp_path, tensors = affine_tensors
    emb = _embed(tmp_path, tensors)
    # The vendored sanitize inserts this key when the checkpoint lacks it;
    # strict load_weights must accept it and it must reach served rows.
    emb.load_weights(
        [("weight_scale", mx.ones((1,), dtype=mx.bfloat16))], strict=True
    )
    emb.load_weights(
        [("weight_scale", mx.full((1,), 2.0, dtype=mx.bfloat16))], strict=True
    )
    ids = [0, 1]
    out = emb(mx.array([ids], dtype=mx.int64)).reshape(2, DIMS)
    expected = _expected_rows(tensors, ids) * 2.0
    assert mx.allclose(out, expected).item()

    rows_before = emb.rows_read
    out2 = emb(mx.array([ids], dtype=mx.int64)).reshape(2, DIMS)
    assert emb.rows_read == rows_before  # scaled values were what got cached
    assert mx.array_equal(out, out2).item()


def test_hot_rows_promote_to_pinned_and_survive_lru_churn(affine_tensors):
    tmp_path, _ = affine_tensors
    emb = _embed(
        tmp_path, None, hot_set_bytes=10 * DIMS * 2, promote_after=3
    )
    assert emb._max_entries == 10
    assert emb.pinned_pool.max_entries == 5

    for _ in range(3):
        emb(mx.array([[0, 1]], dtype=mx.int64))
    assert emb.pinned_pool.contains(0)
    assert emb.pinned_pool.contains(1)

    # Churn the table past the pins; the pinned rows must stay resident and
    # total residency must stay inside the budget.
    emb(mx.array([list(range(2, 14))], dtype=mx.int64))
    assert emb.hot.resident == 10
    assert emb.lru_cache.size == 8
    assert not emb.lru_cache.get(0)
    assert not emb.lru_cache.get(1)

    rows_before = emb.rows_read
    emb(mx.array([[0, 1]], dtype=mx.int64))
    assert emb.rows_read == rows_before  # served from the pinned tier


def test_hot_set_bytes_zero_does_not_crash(affine_tensors):
    tmp_path, _ = affine_tensors
    emb = _embed(tmp_path, None, hot_set_bytes=0, promote_after=2)
    assert emb._max_entries == 1

    idx = mx.array([[0, 1]], dtype=mx.int64)
    for _ in range(3):
        out = emb(idx)
    assert out.shape == (1, 2, DIMS)
    assert emb.hot.resident == 1  # capacity-1 table recycles cleanly


def test_concurrent_requests_interleave_on_engine_thread(affine_tensors):
    """Concurrent requests share one tier on the single engine thread,
    interleaved token-by-token; every call must still land the right rows
    from the right slots."""
    tmp_path, tensors = affine_tensors
    emb = _embed(tmp_path, tensors)
    mid = SHARD_ROWS

    worklists = [
        [0, 3, mid + 2, 7, mid + 2],
        [mid + 5, 1, 1, mid + 9],
        [13, 2, 2, mid, 0],
        [mid + 1, 6, 11, mid + 1],
    ]
    expected = [
        _expected_rows(tensors, flat).reshape(len(flat), DIMS)
        for flat in worklists
    ]

    steps = max(len(flat) for flat in worklists)
    for step in range(steps):
        for req, flat in enumerate(worklists):
            if step >= len(flat):
                continue
            token = flat[step : step + 1]
            out = emb(mx.array([token], dtype=mx.int64)).reshape(1, DIMS)
            assert mx.array_equal(out[0], expected[req][step]).item(), (
                req,
                step,
            )
            assert all(
                0 <= slot < emb.hot.capacity
                for slot in emb.hot._slot_of.values()
            )

    rows_before = emb.rows_read
    for req, flat in enumerate(worklists):
        out = emb(mx.array([flat], dtype=mx.int64)).reshape(len(flat), DIMS)
        assert mx.array_equal(out, expected[req]).item()
    assert emb.rows_read == rows_before


def test_gather_matches_plain_table_with_duplicates(affine_tensors):
    tmp_path, tensors = affine_tensors
    emb = _embed(tmp_path, tensors)
    mid = SHARD_ROWS
    idx = mx.array(
        [[13, 2, 2, mid + 7, 15], [mid + 7, 0, 6, 2, 13]], dtype=mx.int64
    )
    out = emb(idx)
    assert out.shape == (2, 5, DIMS)

    flat = [13, 2, 2, mid + 7, 15, mid + 7, 0, 6, 2, 13]
    expected = _expected_rows(tensors, flat)
    assert mx.array_equal(out.reshape(10, DIMS), expected).item()
    assert emb.last_touched_shards == (0, 1)


def test_read_rows_empty_and_out_of_range(affine_tensors):
    tmp_path, _ = affine_tensors
    emb = _embed(tmp_path, None)
    assert emb.read_rows([]) == []
    with pytest.raises(IndexError):
        emb.read_rows([2 * SHARD_ROWS])
    with pytest.raises(IndexError):
        emb.read_rows([-1])


def _tiny_text_config():
    import mlx_vlm.models.qwen4_exp as qwen4_exp_module

    return qwen4_exp_module.TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=3,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=16,
        moe_intermediate_size=16,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=2,
        max_position_embeddings=128,
        hc_count=2,
        hc_lowrank=8,
        head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        ple_layer_ids=[1],
        ple_embed_dim=32,
        ple_conv_kernel_size=3,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=2,
        eos_token_id=1,
        rope_parameters={
            "rope_type": "default",
            "mrope_section": [2, 1, 1],
            "rope_theta": 10_000,
            "partial_rotary_factor": 1.0,
        },
    )


def _build_mmap_tier(tmp_path, rng_seed: int = 9):
    """Synthetic 4-shard checkpoint + mmap-mode vendored PLE build."""
    from mlx_vlm.models.qwen4_exp.language import (
        Qwen4ExpNGramEmbedding,
        configure_ple_hot_set,
        configure_ple_runtime,
    )

    cfg = _tiny_text_config()
    hf_prefix = "language_model.model.layers.1.ple.ple_embedding.ngram_embedding"
    rng = np.random.default_rng(rng_seed)
    shards = [
        _write_affine_shard(
            tmp_path / f"shard{i}.safetensors",
            22,
            f"{hf_prefix}.shard_{i}",
            rng,
            dims=32,
        )
        for i in range(4)
    ]
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    f"{hf_prefix}.shard_{i}.{key}": f"shard{i}.safetensors"
                    for i in range(4)
                    for key in ("weight", "scales", "biases")
                }
            }
        )
    )
    assert configure_ple_runtime(tmp_path, mode="mmap") == "mmap"
    configure_ple_hot_set(1 << 20)
    return cfg, Qwen4ExpNGramEmbedding(cfg, 128, 1, 0), shards


def test_vendor_mmap_mode_builds_tiered_embedding(tmp_path):
    from mlx_vlm.models.qwen4_exp.language import (
        configure_ple_hot_set,
        configure_ple_runtime,
    )

    try:
        cfg, emb, shards = _build_mmap_tier(tmp_path)
        assert isinstance(emb.ngram_embedding, DiskBackedShardedEmbedding)
        assert emb.ngram_embedding.dims == 32

        ids = [0, 21, 22, 87]
        out = emb(mx.array([ids], dtype=mx.int64), None)
        assert out.shape == (1, 4, 128)
        tier = emb.ngram_embedding
        assert tier.rows_read == 16  # 4 positions x 4 n-gram heads
        rows_before = tier.rows_read
        emb(mx.array([ids], dtype=mx.int64), None)
        assert tier.rows_read == rows_before  # second pass hits the hot set

        full = mx.concatenate(
            [_expected_rows(shard, list(range(22))) for shard in shards], axis=0
        )
        rows = tier(mx.array([ids], dtype=mx.int64)).reshape(4, 32)
        expected = mx.stack([full[i] for i in ids], axis=0)
        assert mx.array_equal(rows, expected).item()
    finally:
        configure_ple_hot_set(None)
        configure_ple_runtime(tmp_path, mode="resident")


def test_configure_ple_hot_set_none_resets_default():
    import mlx_vlm.models.qwen4_exp.language as language

    language.configure_ple_hot_set(123)
    assert language._PLE_HOT_SET_BYTES == 123
    language.configure_ple_hot_set(None)
    assert language._PLE_HOT_SET_BYTES == language._PLE_HOT_SET_BYTES_DEFAULT


def _qsa_indexer():
    """Tiny real QSA indexer: ratio 2, topk 4 -> key_len >= 10 exercises
    the sparse branch."""
    from mlx_vlm.models.qwen3_5.language import Qwen3_5RotaryEmbedding
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpQSAIndexer

    cfg = _tiny_text_config()
    rotary = Qwen3_5RotaryEmbedding(
        int(cfg.head_dim * cfg.rope_parameters["partial_rotary_factor"]),
        max_position_embeddings=cfg.max_position_embeddings,
        base=cfg.rope_parameters["rope_theta"],
        mrope_section=cfg.rope_parameters["mrope_section"],
    )
    return Qwen4ExpQSAIndexer(cfg, rotary)


def test_qsa_indexer_batched_decode_handles_per_row_offsets():
    """Batched caches carry a per-row offset array; the merged indexer cache
    is left-padded to the widest row, so masks use aligned-column semantics."""
    from mlx_vlm.models.qwen4_exp.language import BatchQSAKVCache, QSAKVCache

    indexer = _qsa_indexer()
    hidden = _tiny_text_config().hidden_size

    caches = []
    for length in (10, 8):
        cache = QSAKVCache()
        indexer(
            mx.zeros((1, length, hidden)),
            cache,
            mx.arange(length, dtype=mx.int32)[None],
        )
        kv = mx.zeros((1, 1, length, 4))
        cache.update_and_fetch(kv, kv)
        assert cache.offset == length
        assert cache.index_keys.shape[1] == length
        caches.append(cache)

    merged = BatchQSAKVCache.merge(caches)
    assert merged.offset.tolist() == [10, 8]

    decode_mask = indexer(
        mx.zeros((2, 1, hidden)),
        merged,
        mx.array([[10], [8]], dtype=mx.int32),
    )
    assert decode_mask is not None
    assert decode_mask.shape == (2, 1, 1, 11)
    # Row 0 (offset 10): sparse — 4 of 5 blocks visible + tail token kept.
    assert int(mx.sum(decode_mask[0, 0, 0, :10])) == 8
    assert decode_mask[0, 0, 0, 10].item()
    # Row 1 (offset 8): dense causal window; padded keys stay masked.
    assert mx.all(decode_mask[1, 0, 0, :9]).item()
    assert not mx.any(decode_mask[1, 0, 0, 9:]).item()

    kv = mx.zeros((2, 1, 1, 4))
    merged.update_and_fetch(kv, kv)


def test_qsa_indexer_singleton_decode_mask_unchanged():
    from mlx_vlm.models.qwen4_exp.language import QSAKVCache

    indexer = _qsa_indexer()
    hidden = _tiny_text_config().hidden_size

    cache = QSAKVCache()
    indexer(
        mx.zeros((1, 10, hidden)), cache, mx.arange(10, dtype=mx.int32)[None]
    )
    kv = mx.zeros((1, 1, 10, 4))
    cache.update_and_fetch(kv, kv)

    decode_mask = indexer(
        mx.zeros((1, 1, hidden)), cache, mx.array([[10]], dtype=mx.int32)
    )
    assert decode_mask is not None
    assert decode_mask.shape == (1, 1, 1, 11)
    assert int(mx.sum(decode_mask[0, 0, 0, :10])) == 8
    assert decode_mask[0, 0, 0, 10].item()


def test_batched_qsa_trim_slices_buffers_before_append():
    """trim() rewinds index_offset and slices the physical buffers; the next
    append must land on the trimmed window, not resurrect draft tokens."""
    from mlx_vlm.models.qwen4_exp.language import BatchQSAKVCache, QSAKVCache

    indexer = _qsa_indexer()
    hidden = _tiny_text_config().hidden_size

    caches = []
    for _ in range(2):
        cache = QSAKVCache()
        indexer(
            mx.zeros((1, 10, hidden)), cache, mx.arange(10, dtype=mx.int32)[None]
        )
        kv = mx.zeros((1, 1, 10, 4))
        cache.update_and_fetch(kv, kv)
        caches.append(cache)
    merged = BatchQSAKVCache.merge(caches)

    indexer(mx.zeros((2, 1, hidden)), merged, mx.array([[10], [10]], dtype=mx.int32))
    kv = mx.zeros((2, 1, 1, 4))
    merged.update_and_fetch(kv, kv)
    assert merged.index_offset == 11
    merged.trim(3)
    assert merged.index_offset == 8

    indexer(mx.zeros((2, 1, hidden)), merged, mx.array([[8], [8]], dtype=mx.int32))
    assert merged.index_offset == 9
    assert merged.index_keys.shape[1] == 9
    assert merged.index_position_ids.shape[-1] == 9
