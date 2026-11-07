from math import isclose

import numpy as np
import pytest

import waterz as wz


def test_evaluate() -> None:
    np.random.seed(0)
    seg = np.random.randint(500, size=(3, 3, 3), dtype=np.uint64)
    scores = wz.evaluate(seg, seg)
    assert scores["voi_split"] == 0.0
    assert scores["voi_merge"] == 0.0
    assert scores["rand_split"] == 1.0
    assert scores["rand_merge"] == 1.0

    seg2 = np.random.randint(500, size=(3, 3, 3), dtype=np.uint64)
    scores = wz.evaluate(seg, seg2)

    # print('scores: ', scores)
    # Note that these values are from the first run
    # I have not double checked that this is correct or not.
    # This assertion only make sure that future changes of
    # code will not change the result of the evaluation
    assert isclose(scores["rand_split"], 0.8181818181818182)
    assert isclose(scores["rand_merge"], 0.8709677419354839)
    assert isclose(scores["voi_split"], 0.22222222222222232)
    assert isclose(scores["voi_merge"], 0.14814814814814792)


def test_agglomerate() -> None:
    np.random.seed(0)
    # affinities is a [3,depth,height,width] numpy array of float32
    affinities = np.random.rand(3, 4, 4, 4).astype(np.float32)

    thresholds = [0, 100, 200]
    results = list(wz.agglomerate(affinities, thresholds))
    assert len(results) == 3
    for segmentation in results:
        assert isinstance(segmentation, np.ndarray)
        assert segmentation.shape == (4, 4, 4)
        assert segmentation.dtype == np.uint64
        # just what I observed... from my random test
        # change when better test data is available
        assert np.all(segmentation == 1)


def _shell_case(seed: int):
    rng = np.random.default_rng(seed)
    shape = (12, 14, 16)
    frag = rng.integers(0, 40, size=shape).astype(np.uint32)
    # blocky fragments, so neighbouring voxels repeat ids as real ones do
    frag = np.repeat(np.repeat(frag[:, ::2, ::2], 2, 1), 2, 2)[:, : shape[1], : shape[2]].copy()
    codes = rng.integers(0, 65536, size=(3, *shape)).astype(np.uint16)
    lut = (np.arange(65536, dtype=np.float32) / np.float32(65535)).astype(np.float32)
    sem = rng.integers(0, 5, size=shape).astype(np.uint8)
    low = rng.integers(0, 6, size=(shape[0], (shape[1] + 1) // 2, (shape[2] + 3) // 4)).astype(np.uint64)
    return frag, codes, lut, sem, low


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_agglomerate_shell_matches_agglomerate_on_the_same_values(seed) -> None:
    """Codes through a table, a coarse constraint and a root table: the volume path's result."""
    frag, codes, lut, sem, low = _shell_case(seed)
    affs = lut[codes]
    affs[:, frag == 0] = 0
    for ax in range(3):
        behind = [slice(None)] * 3
        ahead = [slice(None)] * 3
        behind[ax] = slice(None, -1)
        ahead[ax] = slice(1, None)
        affs[(ax, *ahead)][frag[tuple(behind)] == 0] = 0
    z, y, x = np.indices(frag.shape)
    up = low[z, y // 2, x // 4]
    kw = dict(
        semantic_aff_threshold=0.3,
        semantic_size_threshold=5,
        semantic_signal_ratio=0.5,
        semantic_taint_labels=[3],
        semantic_taint_threshold=0.5,
        return_merge_history=True,
        return_region_graph=True,
        return_region_graph_metadata=True,
    )
    # the volume path writes every step into the same fragments buffer: copy as it yields
    ref = [
        (r[0].copy(), *r[1:])
        for r in wz.agglomerate(
            affs=affs, thresholds=[0.0, 0.6], fragments=frag.astype(np.uint64), semantic=sem,
            segconstraint=up, discretize_queue=256, **kw,
        )
    ]
    got = list(
        wz.agglomerate_shell(
            fragments=frag, aff_codes=codes, lut=lut, thresholds=[0.0, 0.6], semantic=sem,
            lowres_constraint=low, constraint_factors=(1, 2, 4), num_nodes=int(frag.max()) + 1,
            **kw,
        )
    )
    for r, g in zip(ref, got):
        np.testing.assert_array_equal(g[0][frag], r[0])
        assert repr(r[1:]) == repr(g[1:])
