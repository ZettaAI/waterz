"""The parallel watershed and shell agglomeration give the serial results, and the opt-in outputs match the default."""

import numpy as np
import pytest

import waterz as wz

LUT = (np.arange(65536, dtype=np.float32) / np.float32(65535)).astype(np.float32)


def _codes(seed: int, shape: tuple) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coarse = rng.random((3, shape[0] // 4 + 1, shape[1] // 4 + 1, shape[2] // 4 + 1))
    field = np.repeat(np.repeat(np.repeat(coarse, 4, 1), 4, 2), 4, 3)[:, : shape[0], : shape[1], : shape[2]]
    field = 0.7 * field + 0.3 * rng.random((3, *shape))
    codes = (field * 65535).astype(np.uint16)
    codes[:, :, :, ::7] //= 3
    return np.ascontiguousarray(codes)


def _shell_case(seed: int, shape: tuple):
    rng = np.random.default_rng(seed)
    blocks = rng.integers(1, 400, size=(shape[0] // 3 + 1, shape[1] // 3 + 1, shape[2] // 3 + 1))
    frag = np.repeat(np.repeat(np.repeat(blocks, 3, 0), 3, 1), 3, 2)[: shape[0], : shape[1], : shape[2]]
    frag = np.ascontiguousarray(np.where(rng.random(shape) < 0.03, 0, frag).astype(np.uint32))
    codes = rng.integers(0, 65536, size=(3, *shape)).astype(np.uint16)
    sem = np.repeat(rng.integers(0, 9, size=(shape[0], shape[1], shape[2] // 4 + 1)), 4, 2)
    sem = np.ascontiguousarray(sem[:, :, : shape[2]].astype(np.uint8))
    low = rng.integers(0, 7, size=(shape[0], (shape[1] + 1) // 2, (shape[2] + 3) // 4)).astype(np.uint64)
    return frag, codes, sem, low


@pytest.fixture
def restore_threads():
    yield
    wz.set_num_threads(0)


@pytest.mark.parametrize("size_threshold", [0, 5, 60])
def test_watershed_codes_is_the_same_at_any_thread_count(size_threshold, restore_threads) -> None:
    codes = _codes(size_threshold, (64, 33, 29))
    wz.set_num_threads(1)
    ref = wz.watershed_codes(codes, LUT, 0.01, 0.98, size_threshold)
    for n in (2, 3, 5, 8):
        wz.set_num_threads(n)
        np.testing.assert_array_equal(wz.watershed_codes(codes, LUT, 0.01, 0.98, size_threshold), ref)


def test_agglomerate_shell_is_the_same_at_any_thread_count(restore_threads) -> None:
    frag, codes, sem, low = _shell_case(7, (40, 30, 28))
    kw = dict(
        fragments=frag, aff_codes=codes, lut=LUT, thresholds=[0.0, 0.6], semantic=sem,
        lowres_constraint=low, constraint_factors=(1, 2, 4), num_nodes=int(frag.max()) + 1,
        semantic_aff_threshold=0.3, semantic_size_threshold=20, semantic_signal_ratio=0.4,
        semantic_taint_labels=[4, 5, 8], semantic_taint_threshold=0.5,
        return_merge_history=True, return_region_graph=True, return_region_graph_metadata=True,
    )
    wz.set_num_threads(1)
    ref = [(r[0].copy(), repr(r[1:])) for r in wz.agglomerate_shell(**kw)]
    for n in (2, 4, 7):
        wz.set_num_threads(n)
        got = [(g[0].copy(), repr(g[1:])) for g in wz.agglomerate_shell(**kw)]
        for (r0, r1), (g0, g1) in zip(ref, got):
            np.testing.assert_array_equal(g0, r0)
            assert g1 == r1


def test_agglomerate_shell_arrays_and_steps_match_the_lists() -> None:
    frag, codes, _, _ = _shell_case(3, (24, 22, 20))
    kw = dict(
        fragments=frag, aff_codes=codes, lut=LUT, thresholds=[0.0, 0.6], num_nodes=int(frag.max()) + 1,
        return_merge_history=True, return_region_graph=True, return_region_graph_metadata=True,
    )
    ref = list(wz.agglomerate_shell(**kw))
    got = list(wz.agglomerate_shell(**kw, merge_history_steps=[1], region_graph_steps=[0], as_arrays=True))
    for r, g in zip(ref, got):
        np.testing.assert_array_equal(g[0], r[0])
    assert got[0][1] is None and got[1][2] is None and got[1][3] is None
    merges = got[1][1]
    assert [(m["a"], m["b"], m["c"], np.float32(m["score"])) for m in ref[1][1]] == [
        (int(m["a"]), int(m["b"]), int(m["c"]), m["score"]) for m in merges
    ]
    edges = got[0][2]
    assert [(e["u"], e["v"], np.float32(e["score"])) for e in ref[0][2]] == [
        (int(e["u"]), int(e["v"]), e["score"]) for e in edges
    ]
    np.testing.assert_array_equal(got[0][3], np.asarray(ref[0][3], dtype=np.float64))
