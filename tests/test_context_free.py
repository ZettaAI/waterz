"""A finished agglomeration leaves nothing behind: its context, constraint providers included, is freed."""

import ctypes
import gc

import numpy as np
import pytest

import waterz as wz


class _MallInfo2(ctypes.Structure):
    _fields_ = [(n, ctypes.c_size_t) for n in (
        "arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks", "fsmblks", "uordblks", "fordblks", "keepcost")]


def _live_bytes():
    libc = ctypes.CDLL("libc.so.6")
    libc.mallinfo2.restype = _MallInfo2
    libc.malloc_trim(0)
    m = libc.mallinfo2()
    return m.uordblks + m.hblkhd


@pytest.mark.skipif(not hasattr(ctypes.CDLL(None), "mallinfo2"), reason="needs glibc's mallinfo2")
def test_repeated_shell_agglomeration_does_not_grow_the_heap() -> None:
    rng = np.random.default_rng(0)
    shape = (40, 60, 60)
    blocks = rng.integers(1, 3000, size=(shape[0] // 2, shape[1] // 2, shape[2] // 2))
    frag = np.ascontiguousarray(np.repeat(np.repeat(np.repeat(blocks, 2, 0), 2, 1), 2, 2).astype(np.uint32))
    codes = rng.integers(0, 65536, size=(3, *shape)).astype(np.uint16)
    lut = (np.arange(65536, dtype=np.float32) / np.float32(65535)).astype(np.float32)
    sem = rng.integers(0, 9, size=shape).astype(np.uint8)
    low = rng.integers(0, 7, size=(shape[0], shape[1] // 2, shape[2] // 4)).astype(np.uint64)

    def run():
        for _ in wz.agglomerate_shell(
            fragments=frag, aff_codes=codes, lut=lut, thresholds=[0.0, 0.6], semantic=sem,
            lowres_constraint=low, constraint_factors=(1, 2, 4), num_nodes=int(frag.max()) + 1,
            semantic_aff_threshold=0.3, semantic_size_threshold=20, semantic_signal_ratio=0.4,
            semantic_taint_labels=[4, 5, 8], semantic_taint_threshold=0.5,
        ):
            pass
        gc.collect()

    run()
    before = _live_bytes()
    for _ in range(5):
        run()
    # each leaked call would keep its providers' per-node maps: ~3,000 nodes x ~380 B
    assert _live_bytes() - before < 200_000
