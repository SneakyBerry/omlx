# SPDX-License-Identifier: Apache-2.0
"""Tiered hot-set overlay for the Qwen3.8-Flash-Next / qwen4_exp PLE table.

A bounded RAM slot table (LRU + promotion to a pinned tier) in front of the
upstream SSD mmap embedding. All row reads — dense BF16/FP8 and affine-packed,
layout validation, ``close()`` — delegate to the vendored
:class:`DiskBackedShardedEmbedding`; this layer only caches already-dequantized
rows in RAM so a hot working set stops re-faulting off SSD under memory
pressure. Placement and caching affect speed, never outputs.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict

import mlx.core as mx
import mlx.nn as nn  # noqa: F401 - re-exported so nn survives lint

from omlx.patches.mlx_vlm_qwen4_exp_compat import apply_mlx_vlm_qwen4_exp_compat_patch

# Ensure the vendored ``mlx_vlm.models.qwen4_exp`` tree is importable before we
# subclass its disk-backed embedding. Idempotent; the vendor module imports oMLX
# lazily, so there is no cycle.
apply_mlx_vlm_qwen4_exp_compat_patch()

from mlx_vlm.models.qwen4_exp.language import (  # noqa: E402
    DiskBackedShardedEmbedding as _DiskBackedShardedEmbedding,
)

logger = logging.getLogger(__name__)


class NGramSlotCache:
    """Bounded hot set in ONE preallocated ``[capacity, dims]`` bf16 table.

    ``capacity`` is the total resident row budget (LRU *and* pinned rows draw
    from one freelist of ``capacity`` slots), so the table never exceeds the
    hot-set byte budget the caller sized it from. Rows are handed out as
    ``mx.take`` copies, never views, so the table stays uniquely referenced and
    every write is the in-place index assignment.

    Concurrency: all inference runs on the single engine stream; table reads and
    in-place writes are stream ops, FIFO-ordered per stream, and each ``__call__``
    drains prior stream work before deciding slots. Metadata is guarded by
    ``_lock``; the table has a single writer (the engine thread).
    """

    def __init__(self, capacity: int, dims: int):
        self.capacity = max(int(capacity), 1)  # total resident rows
        self.dims = dims
        self._table: mx.array | None = None
        self._slot_of: dict[int, int] = {}
        self._lru: OrderedDict[int, None] = OrderedDict()
        self._pinned: OrderedDict[int, None] = OrderedDict()
        self._pinned_cap = max(self.capacity // 2, 1)
        self._free: list[int] = list(range(self.capacity))
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._lock = threading.Lock()

    @property
    def table(self) -> mx.array:
        if self._table is None:
            self._table = mx.zeros((self.capacity, self.dims), dtype=mx.bfloat16)
            # Pin the real buffer now: an in-place update on a still-lazy zeros
            # graph does not survive later re-materialization.
            mx.eval(self._table)
        return self._table

    @property
    def pinned_cap(self) -> int:
        return self._pinned_cap

    @property
    def lru_size(self) -> int:
        with self._lock:
            return len(self._slot_of) - len(self._pinned)

    @property
    def pinned_size(self) -> int:
        with self._lock:
            return len(self._pinned)

    @property
    def resident(self) -> int:
        with self._lock:
            return len(self._slot_of)

    @property
    def hit_rate(self) -> float:
        with self._lock:
            total = self.hits + self.misses
            return (self.hits / total) if total > 0 else 0.0

    def slot_of(self, gram_id: int) -> int | None:
        """Recency-aware lookup; slot index or None on a miss."""
        with self._lock:
            if gram_id in self._pinned:
                self.hits += 1
                return self._slot_of[gram_id]
            if gram_id in self._lru:
                self._lru.move_to_end(gram_id)
                self.hits += 1
                return self._slot_of[gram_id]
            self.misses += 1
            return None

    def _acquire_slot(self) -> int:
        """Pop the freelist, else evict the coldest LRU row, else demote the
        oldest pin. Resident never exceeds ``capacity``. Caller holds _lock."""
        if self._free:
            return self._free.pop()
        if self._lru:
            gram, _ = self._lru.popitem(last=False)
        else:
            gram, _ = self._pinned.popitem(last=False)
        slot = self._slot_of.pop(gram)
        self.evictions += 1
        return slot

    def fill(self, grams, rows: mx.array) -> None:
        """Assign slots to ``grams`` and scatter ``rows`` into the table in one
        in-place write. Already-resident grams are skipped without breaking
        alignment. ``rows`` must not alias the table."""
        with self._lock:
            positions: list[int] = []
            slots: list[int] = []
            for index, gram in enumerate(grams):
                if gram in self._slot_of:
                    continue
                slot = self._acquire_slot()
                self._slot_of[gram] = slot
                self._lru[gram] = None
                positions.append(index)
                slots.append(slot)
            if not positions:
                return
            taken = mx.take(rows, mx.array(positions, dtype=mx.uint32), axis=0)
            table = self.table
            table[mx.array(slots, dtype=mx.uint32)] = taken
            # Force the scatter before any later take of the same buffer: the
            # allocator may recycle the backing storage otherwise.
            mx.eval(table)

    def promote(self, gram_id: int) -> None:
        """Move a resident gram to the pinned tier (demoting the oldest pin when
        full). Slot stays put; no data moves."""
        with self._lock:
            if gram_id in self._pinned or gram_id not in self._slot_of:
                return
            self._lru.pop(gram_id, None)
            if len(self._pinned) >= self._pinned_cap:
                oldest, _ = self._pinned.popitem(last=False)
                self._lru[oldest] = None
            self._pinned[gram_id] = None

    def is_pinned(self, gram_id: int) -> bool:
        with self._lock:
            return gram_id in self._pinned

    def clear(self) -> None:
        with self._lock:
            self._slot_of.clear()
            self._lru.clear()
            self._pinned.clear()
            self._free = list(range(self.capacity))
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            self._table = None


class _CacheStatsView:
    """Read-only view over one tier of an :class:`NGramSlotCache`."""

    def __init__(self, hot: NGramSlotCache, pinned: bool):
        self._hot = hot
        self._pinned = pinned

    @property
    def size(self) -> int:
        return self._hot.pinned_size if self._pinned else self._hot.lru_size

    @property
    def max_entries(self) -> int:
        return self._hot.pinned_cap if self._pinned else self._hot.capacity

    @property
    def hit_rate(self) -> float:
        return self._hot.hit_rate

    def get(self, gram_id: int) -> int | None:
        hot = self._hot
        with hot._lock:
            if self._pinned:
                return hot._slot_of.get(gram_id) if gram_id in hot._pinned else None
            return hot._slot_of.get(gram_id) if gram_id in hot._lru else None

    def contains(self, gram_id: int) -> bool:
        return self._hot.is_pinned(gram_id)

    def clear(self) -> None:
        self._hot.clear()


class DiskBackedShardedEmbedding(_DiskBackedShardedEmbedding):
    """Upstream disk-backed PLE table with a RAM hot-row cache in front."""

    _HOT_SET_BYTES = 2 * 1024 * 1024 * 1024
    _ROW_BYTES = 2  # bf16 dims

    def __init__(
        self,
        model_path,
        prefix: str,
        num_embeddings: int,
        dims: int,
        num_shards: int,
        *,
        hot_set_bytes: int | None = None,
        promote_after: int = 8,
    ):
        super().__init__(model_path, prefix, num_embeddings, dims, num_shards)
        self.hot_set_bytes = (
            int(hot_set_bytes) if hot_set_bytes is not None else self._HOT_SET_BYTES
        )
        # Honest budget: the slot table spans the WHOLE hot set, so
        # ``(lru + pinned) * row_bytes <= hot_set_bytes`` — and never more
        # rows than the checkpoint table itself.
        self._max_entries = min(
            max(self.hot_set_bytes // (dims * self._ROW_BYTES), 1),
            int(num_embeddings),
        )
        self.hot = NGramSlotCache(capacity=self._max_entries, dims=dims)
        self.promote_after = max(int(promote_after), 2)
        self._counts: dict[int, int] = {}
        self._counts_lock = threading.Lock()

    @property
    def lru_cache(self) -> _CacheStatsView:
        return _CacheStatsView(self.hot, pinned=False)

    @property
    def pinned_pool(self) -> _CacheStatsView:
        return _CacheStatsView(self.hot, pinned=True)

    def _bump(self, grams: list[int]) -> None:
        """Count real lookups; pin rows that prove hot. One lock per call."""
        with self._counts_lock:
            promotes = []
            for gram in grams:
                count = self._counts.get(gram, 0) + 1
                self._counts[gram] = count
                if count == self.promote_after:
                    promotes.append(gram)
        for gram in promotes:
            self.hot.promote(gram)

    def read_rows(self, indices: list[int]) -> list[mx.array]:
        """Dequantized rows for ``indices`` straight from the upstream mmap
        path (validated, dtype-correct, weight-scale applied)."""
        if not indices:
            return []
        host = [int(i) for i in indices]
        gathered = super().__call__(mx.array(host, dtype=mx.int64))
        return [row for row in gathered]

    def __call__(self, indices: mx.array) -> mx.array:
        shape = indices.shape
        flat = indices.reshape(-1)
        mx.eval(flat)
        host = [int(i) for i in flat.tolist()]
        if not host:
            return mx.zeros((*shape, self.dims), dtype=mx.bfloat16)

        slots = [self.hot.slot_of(gram) for gram in host]
        hit_positions = [p for p, s in enumerate(slots) if s is not None]
        if len(hit_positions) == len(host):
            rows = mx.take(
                self.hot.table, mx.array(slots, dtype=mx.uint32), axis=0
            )
            self._bump(host)
            return rows.reshape(*shape, self.dims)

        miss_positions = [p for p, s in enumerate(slots) if s is None]
        miss_indices = [host[p] for p in miss_positions]
        miss_batch = super().__call__(mx.array(miss_indices, dtype=mx.int64))
        self.hot.fill(miss_indices, miss_batch)

        out = mx.zeros((len(host), self.dims), dtype=mx.bfloat16)
        if hit_positions:
            hit_rows = mx.take(
                self.hot.table,
                mx.array([slots[p] for p in hit_positions], dtype=mx.uint32),
                axis=0,
            )
            out = out.at[mx.array(hit_positions, dtype=mx.uint32)].add(hit_rows)
        out = out.at[mx.array(miss_positions, dtype=mx.uint32)].add(miss_batch)
        self._bump(host)
        return out.reshape(*shape, self.dims)

    def close(self) -> None:
        self.hot.clear()
        self._counts.clear()
        super().close()
