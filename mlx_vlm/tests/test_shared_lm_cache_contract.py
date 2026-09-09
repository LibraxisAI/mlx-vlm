"""Shared LM/VLM acceptance contracts, authored UNRUN under W1 embargo.

Requires paired U3-LM cache/copy recovery and U4 LM decoder source features.
Stock ab1806 is not a supported substitute; missing methods must fail, not skip.
The LM owner must supply the explicit registration seam and Concatenate fixes.
Only the integrator may collect/run these after explicit W2 closure. No weights,
model constructors, downloads, services, or scheduler monkey-patches are used.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models import cache as lm

from mlx_vlm.models import cache as vlm


def _kv(tokens, cache=None):
    cache = vlm.KVCache() if cache is None else cache
    keys = mx.array(tokens, dtype=mx.float32).reshape(1, 1, len(tokens), 1)
    cache.update_and_fetch(keys, keys + 100)
    return cache


def _history(cache):
    return cache.keys[..., : cache.offset, :].reshape(-1).tolist()


def test_all_ordinary_classes_are_the_shared_objects():
    assert vlm._BaseCache is lm._BaseCache
    assert vlm.KVCache is lm.KVCache
    assert vlm.ArraysCache is lm.ArraysCache
    assert vlm.BatchKVCache is lm.BatchKVCache
    assert vlm.RotatingKVCache is lm.RotatingKVCache
    assert vlm.BatchRotatingKVCache is lm.BatchRotatingKVCache
    assert vlm.CacheList is lm.CacheList
    assert vlm.ChunkedKVCache is lm.ChunkedKVCache
    assert vlm.QuantizedKVCache is lm.QuantizedKVCache
    assert vlm.ConcatenateKVCache is lm.ConcatenateKVCache


def test_paired_lm_prerequisites_are_required_without_fallbacks():
    assert callable(lm.register_cache_type)
    assert vlm.register_cache_type is lm.register_cache_type
    assert callable(lm.KVCache.extract)
    assert callable(lm.QuantizedKVCache.dequantize_for_apc)
    for name in ("prefix_cache_snapshot", "prefix_cache_restore", "prefix_cache_merge"):
        assert callable(getattr(lm._BaseCache, name))
    for cls in (lm.BatchKVCache, lm.BatchRotatingKVCache):
        assert isinstance(cls.batch_size, property)
        assert callable(cls.is_single_row)
    # The owning globals matter: constructors called inside shared methods
    # must not resolve a private VLM duplicate with the same class name.
    assert lm.KVCache.extract.__globals__["KVCache"] is lm.KVCache
    assert lm.BatchKVCache.extract.__globals__["KVCache"] is lm.KVCache
    assert lm.CacheList.extract.__globals__["CacheList"] is lm.CacheList
    assert lm.KVCache.__module__ == "mlx_lm.models.cache"


def test_qwen_dense_and_moe_make_cache_use_shared_classes_without_weights():
    from mlx_vlm.models.qwen3_5.language import LanguageModel as Dense
    from mlx_vlm.models.qwen3_5_moe.language import LanguageModel as Moe

    # Invoke the real methods with only their structural input. No nn.Module
    # construction or model forward is necessary to test cache selection.
    fixture = SimpleNamespace(
        layers=[SimpleNamespace(is_linear=True), SimpleNamespace(is_linear=False)]
    )
    for cls in (Dense, Moe):
        recurrent, attention = cls.make_cache(fixture)
        assert type(recurrent) is lm.ArraysCache
        assert recurrent.state == [None, None]
        assert type(attention) is lm.KVCache
        assert attention.empty()


def test_default_prompt_cache_uses_shared_kv_and_rotating_owners():
    fixture = SimpleNamespace(layers=[object(), object()])
    assert all(type(c) is lm.KVCache for c in vlm.make_prompt_cache(fixture))
    rotating = vlm.make_prompt_cache(fixture, max_kv_size=16)
    assert all(type(c) is lm.RotatingKVCache for c in rotating)
    assert [(c.max_size, c.keep) for c in rotating] == [(16, 4), (16, 4)]


def test_extract_is_a_copy_and_validates_empty_and_negative_indices():
    original = _kv([3, 5, 8])
    extracted = original.extract(-1)
    assert type(extracted) is lm.KVCache
    assert extracted is not original
    extracted.keys[..., 0, :] = -9
    extracted.trim(1)
    assert _history(original) == [3, 5, 8]
    assert _history(extracted) == [-9, 5]
    for index in (1, -2):
        with pytest.raises(IndexError):
            original.extract(index)
    empty = vlm.KVCache()
    for index in (0, -1):
        row = empty.extract(index)
        assert type(row) is lm.KVCache
        assert row is not empty and row.empty() and row.offset == 0
    for index in (1, -2):
        with pytest.raises(IndexError):
            empty.extract(index)


def test_join_filter_and_reextract_preserve_distinct_histories():
    rows = [_kv([1, 2, 3]), _kv([10]), _kv([20, 21])]
    merged = vlm.KVCache.merge(rows)
    assert type(merged) is lm.BatchKVCache
    assert merged.offset.tolist() == [3, 1, 2]
    assert merged.left_padding.tolist() == [0, 2, 1]
    merged.filter(mx.array([2, 0]))
    assert merged.batch_size == 2
    assert not merged.is_single_row()
    for index, expected in enumerate(([20, 21], [1, 2, 3])):
        row = merged.extract(index)
        assert type(row) is lm.KVCache
        assert _history(row) == expected
    nested = vlm.CacheList.merge([vlm.CacheList(rows[0]), vlm.CacheList(rows[1])])
    assert type(nested) is lm.CacheList
    assert type(nested[0]) is lm.BatchKVCache
    extracted = nested.extract(1).extract(0)
    assert type(extracted) is lm.CacheList
    assert type(extracted[0]) is lm.KVCache
    assert _history(extracted[0]) == [10]


def test_right_padding_follows_filtered_rows_before_finalize():
    cache = vlm.BatchKVCache([0, 0])
    cache.prepare(lengths=[1, 3], right_padding=[2, 0])
    keys = mx.array([7, 0, 0, 20, 21, 22], dtype=mx.float32).reshape(2, 1, 3, 1)
    cache.update_and_fetch(keys, keys + 100)
    cache.filter(mx.array([1, 0]))
    assert cache._right_padding.tolist() == [0, 2]
    cache.finalize()
    assert cache.offset.tolist() == [3, 1]
    assert _history(cache.extract(0)) == [20, 21, 22]
    assert _history(cache.extract(1)) == [7]


@pytest.mark.parametrize("populated", [False, True])
def test_arrays_none_slots_and_pending_row_metadata_survive_extraction(populated):
    cache = vlm.ArraysCache(2, left_padding=[2, 0])
    cache.prepare(lengths=[3, 5])
    if populated:
        cache[1] = mx.array([[11, 12], [21, 22]])
    cache.filter(mx.array([1, 0]))
    first, second = cache.extract(0), cache.extract(1)
    assert type(first) is lm.ArraysCache and type(second) is lm.ArraysCache
    assert first[0] is None and second[0] is None
    assert first.left_padding.tolist() == [0]
    assert first.lengths.tolist() == [5]
    assert second.left_padding.tolist() == [2]
    assert second.lengths.tolist() == [3]
    if populated:
        assert first[1].tolist() == [[21, 22]]
        assert second[1].tolist() == [[11, 12]]
        first[1][0, 0] = -1
        assert cache[1].tolist() == [[21, 22], [11, 12]]
    first.left_padding[0] = 9
    first.lengths[0] = 9
    assert cache.left_padding.tolist() == [0, 2]
    assert cache.lengths.tolist() == [5, 3]
    # extend is the existing pending-metadata join; merge is a finalized-row
    # contract and must not be assumed to carry pending metadata in stock ab.
    second.extend(cache.extract(0))
    assert second.left_padding.tolist() == [2, 0]
    assert second.lengths.tolist() == [3, 5]


def test_rotating_last_prefill_token_and_zero_history_merge():
    cache = vlm.BatchRotatingKVCache(4, [0, 0])
    cache.prepare(lengths=[1, 3], right_padding=[2, 0])
    keys = mx.array([7, 0, 20, 21], dtype=mx.float32).reshape(2, 1, 2, 1)
    cache.update_and_fetch(keys, keys + 100)
    cache.filter(mx.array([1, 0]))
    assert cache._lengths.tolist() == [3, 1]
    final = mx.array([22, 0], dtype=mx.float32).reshape(2, 1, 1, 1)
    cache.update_and_fetch(final, final + 100)
    cache.finalize()
    assert cache.offset.tolist() == [3, 1]
    for index, expected in enumerate(([20, 21, 22], [7])):
        row = cache.extract(index)
        assert type(row) is lm.RotatingKVCache
        assert row._temporal_order(row.keys).reshape(-1).tolist() == expected
    empty = _kv([99], vlm.RotatingKVCache(4))
    empty.trim(1)  # allocated backing storage with zero logical history
    full = _kv([4, 5], vlm.RotatingKVCache(4))
    merged = vlm.RotatingKVCache.merge([empty, full])
    assert type(merged) is lm.BatchRotatingKVCache
    assert merged.offset.tolist() == [0, 2]
    assert mx.all(merged.keys[0] == 0).item()
    assert merged.extract(1).keys.reshape(-1).tolist() == [4, 5]


def test_prefix_roundtrip_preserves_ab_state_schema_and_detached_history():
    original = _kv([2, 4, 6])
    snapshot = original.prefix_cache_snapshot()
    assert len(snapshot["state"]) == 2  # ab K/V, not ee19 K/V/offset
    assert snapshot["meta_state"] == ""
    # Snapshot deliberately returns references; detachment belongs to caller.
    snapshot["state"] = tuple(mx.array(x) for x in snapshot["state"])
    restored = vlm.KVCache()
    restored.prefix_cache_restore(snapshot)
    restored.keys[..., 0, :] = 90
    assert _history(original) == [2, 4, 6]
    assert _history(restored) == [90, 4, 6]
    assert restored.prefix_cache_merge([original], [3]) is None
    batched = vlm.KVCache.merge([original, _kv([8])])
    assert len(batched.state) == 4  # K/V, offsets, padding; no ee19 fifth item
    restored_batch = vlm.BatchKVCache.from_state(batched.state, batched.meta_state)
    assert type(restored_batch) is lm.BatchKVCache
    assert _history(restored_batch.extract(1)) == [8]


def test_cache_list_from_state_preserves_nested_specialized_dispatch():
    # The shared owner must resolve registered children through the public API.
    # A local subclass or ignored unknown name cannot satisfy exact identity.
    pooling = vlm.PoolingCache(4)
    pooling.accumulate_windows(
        mx.array([1, 2, 3], dtype=mx.float32).reshape(1, 3, 1),
        mx.array([11, 12, 13], dtype=mx.float32).reshape(1, 3, 1),
        offset=0,
    )
    original = vlm.CacheList(
        _kv([5, 6]),
        vlm.CacheList(pooling, _kv([8, 9], vlm.ConcatenateKVCache())),
    )
    restored = vlm.CacheList.from_state(original.state, original.meta_state)
    assert type(restored) is lm.CacheList
    assert type(restored[0]) is lm.KVCache
    assert _history(restored[0]) == [5, 6]
    assert type(restored[1]) is lm.CacheList
    assert type(restored[1][0]) is vlm.PoolingCache
    assert restored[1][0].ratio == 4 and restored[1][0].remainder == 3
    assert restored[1][0].state[0].reshape(-1).tolist() == [1, 2, 3]
    assert type(restored[1][1]) is lm.ConcatenateKVCache
    assert _history(restored[1][1]) == [8, 9]


def test_apc_recursive_snapshot_keeps_pooling_and_canonical_kv_independent():
    from mlx_vlm.apc import snapshot_prompt_cache_row
    from mlx_vlm.apc_adapters import merge_cache_entries

    pooling = vlm.PoolingCache(4)
    pooling.accumulate_windows(
        mx.array([1, 2, 3], dtype=mx.float32).reshape(1, 3, 1),
        mx.ones((1, 3, 1)),
        offset=0,
    )
    original = vlm.CacheList(_kv([7, 8, 9]), pooling)
    snapshot = snapshot_prompt_cache_row([original], batch_idx=0)
    assert snapshot is not None
    assert type(snapshot[0]) is lm.CacheList
    assert type(snapshot[0][0]) is lm.KVCache
    assert type(snapshot[0][1]) is vlm.PoolingCache
    pooling.buf_kv[0, 0, 0] = 99
    assert snapshot[0][1].state[0].reshape(-1).tolist() == [1, 2, 3]
    merged = merge_cache_entries([snapshot[0][1], vlm.PoolingCache(4)], [3, 0])
    assert type(merged) is vlm.BatchPoolingCache
    assert merged.remainder == [3, 0]
    assert merged._processed == [3, 0]
    restored = merged.extract(0)
    assert type(restored) is vlm.PoolingCache
    assert restored.state[0].reshape(-1).tolist() == [1, 2, 3]


@pytest.mark.parametrize("bits", [4, 8])
def test_quantized_apc_and_batch_extraction_return_canonical_owner(bits):
    # Constant groups give independent exact expected values at either width.
    keys = mx.full((2, 1, 3, 32), 2.0)
    values = mx.full((2, 1, 3, 32), 7.0)
    batch = vlm.BatchQuantizedKVCache([1, 0], group_size=32, bits=bits)
    batch.update_and_fetch(keys, values)
    batch.filter(mx.array([0]))
    row = batch.extract(0)
    assert type(row) is lm.QuantizedKVCache
    assert row.offset == 2
    dk, dv = row.dequantize_for_apc()
    assert dk.shape == dv.shape == (1, 1, 2, 32)
    assert (
        mx.allclose(dk, mx.full(dk.shape, 2.0)).item()
        and mx.allclose(dv, mx.full(dv.shape, 7.0)).item()
    )
    bk, bv = batch.dequantize_for_apc()
    assert mx.array_equal(dk, bk).item() and mx.array_equal(dv, bv).item()
    assert vlm.QuantizedKVCache().dequantize_for_apc() == (None, None)
    assert vlm.BatchQuantizedKVCache([0]).dequantize_for_apc() == (None, None)
    converted = _kv([1, 2])
    converted.keys = mx.full((1, 1, 2, 32), 3.0)
    converted.values = mx.full((1, 1, 2, 32), 9.0)
    quantized = converted.to_quantized(group_size=32, bits=bits)
    assert type(quantized) is lm.QuantizedKVCache
    qk, qv = quantized.dequantize_for_apc()
    assert (
        mx.allclose(qk, mx.full(qk.shape, 3.0)).item()
        and mx.allclose(qv, mx.full(qv.shape, 9.0)).item()
    )


def test_specialized_algorithms_keep_vlm_identity_and_public_helpers():
    for cls in (
        vlm.BufferedRotatingKVCache,
        vlm.BatchQuantizedKVCache,
        vlm.PoolingCache,
        vlm.BatchPoolingCache,
        vlm.SimpleKVCache,
        vlm.StaticPrefixKVCache,
    ):
        assert cls.__module__ == "mlx_vlm.models.cache"
    buffered = vlm.BufferedRotatingKVCache.from_cache(
        _kv([2, 3, 4], vlm.RotatingKVCache(4)), buffer_size=2
    )
    _kv([5, 6], buffered)
    assert buffered.trim(2) == 2
    assert buffered.state[0].reshape(-1).tolist() == [2, 3, 4]
    prefix = _kv([10, 20], vlm.StaticPrefixKVCache(4))
    borrowed = vlm.StaticPrefixKVCache.from_prefix(prefix)
    key = mx.array([30], dtype=mx.float32).reshape(1, 1, 1, 1)
    visible, _ = borrowed.update_and_fetch(key, key + 100)
    assert visible.reshape(-1).tolist() == [10, 20, 30]
    assert borrowed.offset == prefix.offset == 2
    assert [vlm.should_quantize_kv_layer(i, 3) for i in range(3)] == [True, True, False]
    assert vlm.create_causal_mask(2).tolist() == [[True, False], [True, True]]


def test_concatenate_empty_restore_and_trim_append_keep_exact_history():
    empty = vlm.ConcatenateKVCache.from_state((None, None), "")
    assert type(empty) is lm.ConcatenateKVCache
    assert empty.empty() and empty.offset == 0
    cache = _kv([1, 2, 3, 4], empty)
    assert cache.trim(2) == 2
    assert cache.keys.reshape(-1).tolist() == [1, 2]
    assert cache.values.reshape(-1).tolist() == [101, 102]
    _kv([9], cache)
    assert _history(cache) == [1, 2, 9]
    assert cache.values.reshape(-1).tolist() == [101, 102, 109]
    assert cache.trim(99) == 3
    _kv([7], cache)
    assert _history(cache) == [7]
    assert cache.values.reshape(-1).tolist() == [107]


def test_specialized_registration_is_idempotent_and_rejects_new_generations():
    for cls in (
        vlm.BufferedRotatingKVCache,
        vlm.BatchQuantizedKVCache,
        vlm.PoolingCache,
        vlm.BatchPoolingCache,
        vlm.StaticPrefixKVCache,
    ):
        assert issubclass(cls, lm._BaseCache)
        assert callable(cls.from_state)
        # Successful import has already registered these exact objects.
        lm.register_cache_type(cls)
        lm.register_cache_type(cls)
        replacement = type(cls.__name__, (lm._BaseCache,), {})
        with pytest.raises(ValueError, match="already registered"):
            lm.register_cache_type(replacement)
    restored = lm.CacheList.from_state([(None, None, None)], (["PoolingCache"], [4]))
    assert type(restored[0]) is vlm.PoolingCache
    assert restored[0].empty()
    assert "PoolingCache" not in vars(lm)  # no VLM injection into LM globals


def test_codec_registration_rejects_noncanonical_owners_and_reserved_names():
    class ForeignBase:
        @classmethod
        def from_state(cls, state, meta_state):
            return cls()

    class MissingDecoder(lm._BaseCache):
        from_state = None

    for invalid in (ForeignBase, MissingDecoder, vlm.SimpleKVCache, lm._BaseCache, 7):
        with pytest.raises(TypeError, match="_BaseCache subclass"):
            lm.register_cache_type(invalid)
    for name in ("KVCache", "ConcatenateKVCache", "load_prompt_cache", "list"):
        with pytest.raises(ValueError, match="reserved"):
            lm.register_cache_type(type(name, (lm._BaseCache,), {}))
    with pytest.raises(ValueError, match="reserved"):
        lm.register_cache_type(lm.KVCache)
    assert vlm.KVCache is lm.KVCache
    for name in ("ForeignBase", "MissingDecoder", "SimpleKVCache", "load_prompt_cache"):
        with pytest.raises(ValueError, match="Unknown cache type"):
            vlm.CacheList.from_state([[]], ([name], [""]))


@pytest.mark.parametrize(
    "name", ["MissingVLMCodec", "mlx_vlm.models.cache.PoolingCache", 7]
)
def test_public_decoder_rejects_unknown_or_nonstring_names(name):
    with pytest.raises(ValueError, match="Unknown cache type|must be a string"):
        vlm.CacheList.from_state([[]], ([name], [""]))


@pytest.mark.parametrize(
    "state, metadata",
    [([], (["KVCache"], [""])), ([[]], ([], [""])), ([[]], (["KVCache"], []))],
)
def test_public_decoder_rejects_parallel_arity_mismatch(state, metadata):
    with pytest.raises(ValueError, match="equal lengths"):
        vlm.CacheList.from_state(state, metadata)


def test_registered_buffered_restore_keeps_rollback_and_append_history():
    original = _kv([2, 3, 4], vlm.BufferedRotatingKVCache(4, buffer_size=2))
    restored = vlm.CacheList.from_state(
        [original.state], (["BufferedRotatingKVCache"], [original.meta_state])
    )[0]
    assert type(restored) is vlm.BufferedRotatingKVCache
    assert restored.meta_state == original.meta_state
    assert restored.trim(1) == 1
    _kv([9], restored)
    assert restored.state[0].reshape(-1).tolist() == [2, 3, 9]


def test_registered_batch_quantized_restore_can_filter_and_finalize():
    original = vlm.BatchQuantizedKVCache([0, 0], group_size=32, bits=4)
    original.update_and_fetch(mx.full((2, 1, 2, 32), 2.0), mx.full((2, 1, 2, 32), 7.0))
    restored = vlm.CacheList.from_state(
        [original.state], (["BatchQuantizedKVCache"], [original.meta_state])
    )[0]
    assert type(restored) is vlm.BatchQuantizedKVCache
    restored.filter(mx.array([1]))
    restored.finalize()
    row = restored.extract(0)
    assert type(row) is lm.QuantizedKVCache and row.offset == 2
    keys, values = row.dequantize_for_apc()
    assert (
        mx.allclose(keys, mx.full(keys.shape, 2.0)).item()
        and mx.allclose(values, mx.full(values.shape, 7.0)).item()
    )


def test_registered_batch_pooling_restore_can_resume_and_filter():
    original = vlm.BatchPoolingCache(4, [0, 0])
    kv = mx.array([1, 2, 10, 20], dtype=mx.float32).reshape(2, 2, 1)
    original.accumulate_windows(kv, kv + 100, offset=0)
    restored = vlm.CacheList.from_state(
        [original.state], (["BatchPoolingCache"], [original.meta_state])
    )[0]
    assert type(restored) is vlm.BatchPoolingCache
    restored.filter(mx.array([1]))
    kv = mx.array([30], dtype=mx.float32).reshape(1, 1, 1)
    restored.accumulate_windows(kv, kv + 100, offset=2)
    row = restored.extract(0)
    assert row.remainder == 3
    assert row.state[0].reshape(-1).tolist() == [10, 20, 30]


def test_registered_static_prefix_restore_and_borrow_keep_separate_contracts():
    original = _kv([1, 2], vlm.StaticPrefixKVCache(8))
    restored = vlm.CacheList.from_state(
        [original.state], (["StaticPrefixKVCache"], [original.meta_state])
    )[0]
    assert type(restored) is vlm.StaticPrefixKVCache
    assert not restored.read_only
    assert restored.meta_state == original.meta_state
    _kv([3], restored)
    assert _history(restored) == [1, 2, 3]
    borrowed = vlm.StaticPrefixKVCache.from_prefix(restored)
    keys = mx.array([9], dtype=mx.float32).reshape(1, 1, 1, 1)
    visible, _ = borrowed.update_and_fetch(keys, keys + 100)
    assert visible.reshape(-1).tolist() == [1, 2, 3, 9]
    assert borrowed.read_only and borrowed.offset == restored.offset == 3


def test_registered_specializations_restore_empty_raw_state():
    originals = (
        vlm.BufferedRotatingKVCache(4),
        vlm.BatchQuantizedKVCache([0]),
        vlm.PoolingCache(4),
        vlm.BatchPoolingCache(4, [0]),
        vlm.StaticPrefixKVCache(4),
    )
    for original in originals:
        restored = lm.CacheList.from_state(
            [original.state], ([type(original).__name__], [original.meta_state])
        )[0]
        assert type(restored) is type(original)
        assert restored.empty()
    assert not issubclass(vlm.SimpleKVCache, lm._BaseCache)
    assert not hasattr(vlm.SimpleKVCache, "from_state")


def test_file_loader_uses_registered_vlm_decoder_and_canonical_nested_owners(tmp_path):
    buffered = _kv([4, 5], vlm.BufferedRotatingKVCache(8, buffer_size=2))
    original = vlm.CacheList(_kv([1, 2]), buffered, _kv([7], vlm.ConcatenateKVCache()))
    path = str(tmp_path / "shared-cache.safetensors")
    lm.save_prompt_cache(path, [original], metadata={"purpose": "codec-contract"})
    restored, metadata = lm.load_prompt_cache(path, return_metadata=True)
    assert metadata == {"purpose": "codec-contract"}
    assert type(restored[0]) is lm.CacheList
    assert type(restored[0][0]) is lm.KVCache
    assert type(restored[0][1]) is vlm.BufferedRotatingKVCache
    assert type(restored[0][2]) is lm.ConcatenateKVCache
    assert _history(restored[0][0]) == [1, 2]
    assert restored[0][1].state[0].reshape(-1).tolist() == [4, 5]
    assert _history(restored[0][2]) == [7]
