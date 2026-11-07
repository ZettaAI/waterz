"""JIT-compiled watershed + agglomeration on affinity graphs.

Supports both float32 and uint8 affinities.  Each (dtype, scoring_function,
discretize_queue) combination gets its own cached compiled module via witty.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from types import ModuleType

    from numpy.typing import NDArray

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# JIT compilation helper
# ---------------------------------------------------------------------------

_AFF_DTYPE_MAP = {
    np.dtype("float32"): ("float", "float"),
    np.dtype("uint8"): ("uint8_t", "uint8_t"),
}

_SEG_DTYPE_MAP = {
    np.dtype("uint64"): "uint64_t",
    np.dtype("uint32"): "uint32_t",
}

# Cython source patches for uint8 mode.
# Each (old, new) pair is applied via str.replace.
_PYX_UINT8_PATCHES = [
    # Typed array declarations
    ("np.float32_t, ndim=4", "np.uint8_t, ndim=4"),
    # Local pointer declarations
    ("cdef float*    aff_data", "cdef uint8_t*  aff_data"),
    ("cdef float* aff_data", "cdef uint8_t* aff_data"),
    # C extern function signatures (affinities)
    ("const float*    affinity_data", "const uint8_t*  affinity_data"),
    ("const float*            affinity_data", "const uint8_t*          affinity_data"),
    # Struct score fields + threshold
    ("float score", "uint8_t score"),
    ("float           affThresholdLow", "uint8_t         affThresholdLow"),
    ("float           affThresholdHigh", "uint8_t         affThresholdHigh"),
    ("float        threshold", "uint8_t      threshold"),
]

# Cython source patches for uint32 segmentation.
_PYX_UINT32_SEG_PATCHES = [
    # Typed array declarations (segmentation / fragments)
    ("uint64_t, ndim=3]     segmentation", "uint32_t, ndim=3]     segmentation"),
    ("uint64_t, ndim=3] seg", "uint32_t, ndim=3] seg"),
    # Local pointer declarations
    ("cdef uint64_t* segmentation_data", "cdef uint32_t* segmentation_data"),
    ("cdef uint64_t* seg_data", "cdef uint32_t* seg_data"),
    # Array creation dtype
    ("dtype=np.uint64", "dtype=np.uint32"),
    # C extern: struct fields (Merge, ScoredEdge use SegID = uint32_t)
    ("        uint64_t a\n        uint64_t b\n        uint64_t c",
     "        uint32_t a\n        uint32_t b\n        uint32_t c"),
    ("        uint64_t u\n        uint64_t v",
     "        uint32_t u\n        uint32_t v"),
    # C extern: function signatures
    ("            uint64_t*       segmentation_data,", "            uint32_t*       segmentation_data,"),
    ("            uint64_t*       segmentation_data)", "            uint32_t*       segmentation_data)"),
]

_SOURCE_KEY_GLOBS = (
    '_agglomerate.py',
    'agglomerate.pyx',
    'frontend_agglomerate.cpp',
    'frontend_agglomerate.h',
    'backend/**/*.h',
    'backend/**/*.hpp',
    'backend/**/*.cpp',
)


def _source_fingerprint() -> str:
    """Hash local JIT sources so ABI changes invalidate cached builds."""
    digest = hashlib.md5()
    for pattern in _SOURCE_KEY_GLOBS:
        for path in sorted(HERE.glob(pattern)):
            if not path.is_file():
                continue
            digest.update(str(path.relative_to(HERE)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def _compile_module(
    scoring_function: str,
    discretize_queue: int = 0,
    aff_dtype: np.dtype = np.dtype("float32"),
    seg_dtype: np.dtype = np.dtype("uint64"),
    force_rebuild: bool = False,
) -> "ModuleType":
    """Compile and cache the agglomerate Cython/C++ module.

    Returns the compiled module with ``agglomerate()`` and
    ``buildRegionGraphOnly()`` entry points.
    """
    import fcntl
    import os
    import subprocess
    import sys as _sys

    import witty

    if aff_dtype not in _AFF_DTYPE_MAP:
        raise ValueError(f"Unsupported aff_dtype {aff_dtype}; expected float32 or uint8")
    aff_ctype, score_ctype = _AFF_DTYPE_MAP[aff_dtype]

    if seg_dtype not in _SEG_DTYPE_MAP:
        raise ValueError(f"Unsupported seg_dtype {seg_dtype}; expected uint64 or uint32")
    seg_ctype = _SEG_DTYPE_MAP[seg_dtype]

    # Cache keys must reflect source/header ABI, not just runtime parameters.
    cache_dir = witty.get_witty_cache_dir()
    source_key = _source_fingerprint()
    header_key = hashlib.md5(
        f"{scoring_function}|{discretize_queue}|{aff_ctype}|{seg_ctype}|{source_key}".encode()
    ).hexdigest()[:12]
    header_dir = cache_dir / f"_waterz_headers_{header_key}"
    header_dir.mkdir(parents=True, exist_ok=True)

    # SegType.h — parameterises SegID (uint64_t or uint32_t)
    (header_dir / "SegType.h").write_text(
        f"typedef {seg_ctype} SegID;\n"
    )

    # AffType.h — parameterises AffValue and ScoreValue
    (header_dir / "AffType.h").write_text(
        f"typedef {aff_ctype} AffValue;\n"
        f"typedef {score_ctype} ScoreValue;\n"
    )

    # ScoringFunction.h
    (header_dir / "ScoringFunction.h").write_text(
        f"typedef {scoring_function} ScoringFunctionType;"
    )

    # Queue.h
    queue_src = "template<typename T, typename S> using QueueType = " + (
        "PriorityQueue<T, S>;"
        if discretize_queue == 0
        else f"BinQueue<T, S, {discretize_queue}>;"
    )
    (header_dir / "Queue.h").write_text(queue_src)

    include_dirs = [
        str(HERE),
        str(header_dir),
        str(HERE / "backend"),
        np.get_include(),
    ]
    _boost_inc = Path(_sys.prefix) / "include"
    if (_boost_inc / "boost").is_dir():
        include_dirs.append(str(_boost_inc))

    _compile_args = ["-std=c++17", "-w"]

    # File lock to prevent concurrent JIT compilation across workers
    lock_path = cache_dir / f"_waterz_compile_{header_key}.lock"
    lock_file = open(lock_path, "w")
    fcntl.flock(lock_file, fcntl.LOCK_EX)
    try:
        # Pre-compile frontend C++
        frontend_cpp = HERE / "frontend_agglomerate.cpp"
        obj_path = cache_dir / f"_waterz_frontend_{header_key}.o"

        if not obj_path.exists() or force_rebuild:
            compile_cmd = ["g++", "-c", "-fPIC", "-std=c++17", "-w", "-O2"]
            for d in include_dirs:
                compile_cmd += ["-I", d]
            compile_cmd += [str(frontend_cpp), "-o", str(obj_path)]
            env = os.environ.copy()
            env["CCACHE_DISABLE"] = "1"
            subprocess.check_call(compile_cmd, env=env)

        # Patch .pyx source for uint8 affinities and/or uint32 segmentation
        pyx_source = (HERE / "agglomerate.pyx").read_text()
        if aff_ctype == "uint8_t":
            for old, new in _PYX_UINT8_PATCHES:
                pyx_source = pyx_source.replace(old, new)
        if seg_ctype == "uint32_t":
            for old, new in _PYX_UINT32_SEG_PATCHES:
                pyx_source = pyx_source.replace(old, new)

        def _build_frontend(cache_dir_: Path) -> list[str]:
            return [str(obj_path)]

        module = witty.compile_cython(
            pyx_source,
            source_files=[str(frontend_cpp)],
            build_extra_objects=_build_frontend,
            extra_link_args=["-std=c++17"],
            extra_compile_args=_compile_args,
            include_dirs=include_dirs,
            language="c++",
            quiet=True,
            force_rebuild=force_rebuild,
        )
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()
    return module


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_DEFAULT_SCORING = "OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>"

# (scoring_function, discretize_queue, aff_dtype, seg_dtype) -> shipped module name.
# These variants are pre-compiled in setup.py, so workers import a ready .so
# instead of JIT-compiling at runtime (which can race across processes and
# corrupt the witty cache). Combinations not listed fall back to JIT.
_PRECOMPILED_MODULES = {
    (_DEFAULT_SCORING, 0, np.dtype("float32"), np.dtype("uint64")): "_agglomerate_default",
    (_DEFAULT_SCORING, 256, np.dtype("float32"), np.dtype("uint64")): "_agglomerate_mean_bin256",
}


def agglomerate_shell(
    fragments: NDArray[np.uint32],
    aff_codes: NDArray,
    lut: NDArray[np.float32],
    thresholds: Sequence[float],
    semantic: NDArray[np.uint8] | None = None,
    lowres_constraint: NDArray[np.uint64] | None = None,
    constraint_factors: tuple[int, int, int] = (1, 1, 1),
    constraint_offsets: tuple[int, int, int] = (0, 0, 0),
    node_constraint: NDArray[np.uint64] | None = None,
    num_nodes: int | None = None,
    return_merge_history: bool = False,
    return_region_graph: bool = False,
    return_region_graph_metadata: bool = False,
    semantic_aff_threshold: float = 0.5,
    semantic_size_threshold: int = 100_000,
    semantic_signal_ratio: float = 0.6,
    semantic_taint_labels: list[int] = [],
    semantic_taint_threshold: float = 0.0,
    size_heuristic_aff_threshold: float = 1.0,
    size_heuristic_small_threshold: int = 1_000_000,
    size_heuristic_large_threshold: int = 10_000_000,
    discretize_queue: int = 256,
) -> Iterator[tuple]:
    """Agglomerate from the shell: compact uint32 ``fragments`` (``Z Y X``), affinity codes
    (uint8 or uint16, ``(3, Z, Y, X)``, channels z, y, x) read through ``lut``, and the
    segmentation constraint as a coarse grid (voxel ``v`` reads
    ``lowres[(v + offsets) // factors]``) or one id per node.

    Gives the merges, region graph and roots that :func:`agglomerate` gives on
    ``lut[aff_codes]`` with every face touching fragment 0 set to 0, without either volume
    being built. Yields ``(roots, ...)`` per threshold; ``roots[n]`` is fragment ``n``'s node.
    """
    import importlib

    name = _PRECOMPILED_MODULES.get(
        (_DEFAULT_SCORING, discretize_queue, np.dtype("float32"), np.dtype("uint64"))
    )
    if name is None:
        raise ValueError(f"no precompiled module for discretize_queue={discretize_queue}")
    module = importlib.import_module(f"waterz.{name}")
    n = int(fragments.max()) + 1 if num_nodes is None else int(num_nodes)
    return module.agglomerate_shell(
        fragments, n, aff_codes, np.ascontiguousarray(lut, dtype=np.float32), list(thresholds),
        semantic, lowres_constraint, tuple(int(v) for v in constraint_factors),
        tuple(int(v) for v in constraint_offsets), node_constraint,
        semantic_aff_threshold, semantic_size_threshold, semantic_signal_ratio,
        list(semantic_taint_labels), semantic_taint_threshold,
        size_heuristic_aff_threshold, size_heuristic_small_threshold,
        size_heuristic_large_threshold,
        return_merge_history, return_region_graph, return_region_graph_metadata,
    )


def agglomerate(
    affs: NDArray[np.float32] | None = None,
    thresholds: Sequence[float] | None = None,
    gt: NDArray[np.uint32] | None = None,
    fragments: NDArray[np.uint64] | None = None,
    semantic: NDArray[np.uint8] | None = None,
    segconstraint: NDArray[np.uint64] | None = None,
    input_rag=None,
    input_rag_metadata=None,
    aff_threshold_low: float = 0.0001,
    aff_threshold_high: float = 0.9999,
    return_merge_history: bool = False,
    return_region_graph: bool = False,
    return_region_graph_metadata: bool = False,
    scoring_function: str = _DEFAULT_SCORING,
    semantic_aff_threshold: float = 0.5,
    semantic_size_threshold: int = 100_000,
    semantic_signal_ratio: float = 0.6,
    semantic_taint_labels: list[int] = [],
    semantic_taint_threshold: float = 0.0,
    size_heuristic_aff_threshold: float = 1.0,
    size_heuristic_small_threshold: int = 1_000_000,
    size_heuristic_large_threshold: int = 10_000_000,
    discretize_queue: int = 0,
    seg_dtype: np.dtype | str = "uint64",
    force_rebuild: bool = False,
) -> Iterator[tuple | NDArray[np.uint64]]:
    """Compute segmentations from an affinity graph for several thresholds.

    Accepts **float32** or **uint8** affinities.  When uint8 is provided,
    the entire C++ pipeline operates on uint8 — no conversion to float32.

    Parameters
    ----------

        affs: numpy array, float32 or uint8, 4 dimensional

            The affinities as an array with affs[channel][z][y][x].

        thresholds: list of float

            The thresholds to compute segmentations for. For each threshold, one
            segmentation is returned.

        gt: numpy array, uint32, 3 dimensional (optional)

            An optional ground-truth segmentation as an array with gt[z][y][x].
            If given, metrics

        fragments: numpy array, uint64, 3 dimensional (optional)

            An optional volume of fragments to use, instead of the build-in
            zwatershed.

        aff_threshold_low: float, default 0.0001
        aff_threshold_high: float, default 0.9999,

            Thresholds on the affinities for the initial segmentation step.

        return_merge_history: bool

            If set to True, the returning tuple will contain a merge history,
            relative to the previous segmentation.

        return_region_graph: bool

            If set to True, the returning tuple will contain the region graph
            for the returned segmentation.

        scoring_function: string, default 'OneMinus<MeanAffinity<RegionGraphType, ScoreValue>>'

            A C++ type string specifying the edge scoring function to use.

        discretize_queue: int

            If set to non-zero, a bin queue with that many bins will be used to
            approximate the priority queue for merge operations.

        seg_dtype: dtype, default 'uint64'

            Segmentation dtype (uint64 or uint32).

        force_rebuild: bool

            Force the rebuild of the module. Only needed for development.

    Returns
    -------

        Results are returned as tuples from a generator object.
    """
    # Determine whether to use pre-compiled default module
    if affs is not None:
        affs = np.ascontiguousarray(affs)
        aff_dtype = affs.dtype
        if aff_dtype == np.float64:
            affs = affs.astype(np.float32)
            aff_dtype = np.dtype("float32")
    else:
        aff_dtype = np.dtype("float32")

    seg_dtype = np.dtype(seg_dtype)

    _precompiled_name = _PRECOMPILED_MODULES.get(
        (scoring_function, discretize_queue, aff_dtype, seg_dtype)
    )

    if _precompiled_name is not None:
        import importlib

        module = importlib.import_module(f"waterz.{_precompiled_name}")
    else:
        if aff_dtype not in _AFF_DTYPE_MAP:
            raise TypeError(
                f"affs.dtype must be float32 or uint8, got {aff_dtype}"
            )
        module = _compile_module(
            scoring_function, discretize_queue,
            aff_dtype=aff_dtype, seg_dtype=seg_dtype, force_rebuild=force_rebuild,
        )

    # call compiled function
    if input_rag is not None or input_rag_metadata is not None:
        return module.agglomerate_rag(
            rag=input_rag,
            rag_metadata=input_rag_metadata,
            thresholds=thresholds,
            fragments=fragments,
        )

    return module.agglomerate(
        affs=affs,
        thresholds=thresholds,
        gt=gt,
        fragments=fragments,
        semantic=semantic,
        segconstraint=segconstraint,
        aff_threshold_low=aff_threshold_low,
        aff_threshold_high=aff_threshold_high,
        semantic_aff_threshold=semantic_aff_threshold,
        semantic_size_threshold=semantic_size_threshold,
        semantic_signal_ratio=semantic_signal_ratio,
        semantic_taint_labels=semantic_taint_labels,
        semantic_taint_threshold=semantic_taint_threshold,
        size_heuristic_aff_threshold=size_heuristic_aff_threshold,
        size_heuristic_small_threshold=size_heuristic_small_threshold,
        size_heuristic_large_threshold=size_heuristic_large_threshold,
        return_merge_history=return_merge_history,
        return_region_graph=return_region_graph,
        return_region_graph_metadata=return_region_graph_metadata,
    )


def waterz(
    affs: NDArray,
    thresholds: Sequence[float],
    gt: NDArray[np.uint32] | None = None,
    fragments: NDArray[np.uint64] | None = None,
    aff_threshold_low: float = 0.0001,
    aff_threshold_high: float = 0.9999,
    return_merge_history: bool = False,
    return_region_graph: bool = False,
    scoring_function: str = _DEFAULT_SCORING,
    discretize_queue: int = 0,
    seg_dtype: np.dtype | str = "uint64",
    force_rebuild: bool = False,
) -> Iterator[tuple | NDArray]:
    """Thin wrapper matching the pytorch/waterz API signature."""
    return agglomerate(
        affs=affs,
        thresholds=thresholds,
        gt=gt,
        fragments=fragments,
        aff_threshold_low=aff_threshold_low,
        aff_threshold_high=aff_threshold_high,
        return_merge_history=return_merge_history,
        return_region_graph=return_region_graph,
        scoring_function=scoring_function,
        discretize_queue=discretize_queue,
        seg_dtype=seg_dtype,
        force_rebuild=force_rebuild,
    )


def build_region_graph_only(
    affs: NDArray,
    fragments: NDArray,
    scoring_function: str = "MeanAffinity<RegionGraphType, ScoreValue>",
    discretize_queue: int = 0,
    seg_dtype: np.dtype | str = "uint64",
    force_rebuild: bool = False,
) -> list:
    """Build scored region graph without RegionMerging or priority queue.

    Accepts float32 or uint8 affinities, uint64 or uint32 segmentation.

    Returns list of dicts ``[{"u": id1, "v": id2, "score": float}, ...]``.
    """
    affs = np.ascontiguousarray(affs)
    aff_dtype = affs.dtype
    if aff_dtype == np.float64:
        affs = affs.astype(np.float32)
        aff_dtype = np.dtype("float32")
    if aff_dtype not in _AFF_DTYPE_MAP:
        raise TypeError(f"affs.dtype must be float32 or uint8, got {aff_dtype}")

    seg_dtype = np.dtype(seg_dtype)
    fragments = np.ascontiguousarray(fragments, dtype=seg_dtype)

    module = _compile_module(
        scoring_function, discretize_queue,
        aff_dtype=aff_dtype, seg_dtype=seg_dtype, force_rebuild=force_rebuild,
    )

    return module.buildRegionGraphOnly(affs, fragments)


def build_region_graph_rich(
    affs: NDArray,
    fragments: NDArray,
    scoring_function: str = "MeanAffinity<RegionGraphType, ScoreValue>",
    discretize_queue: int = 0,
    seg_dtype: np.dtype | str = "uint64",
    force_rebuild: bool = False,
) -> list:
    """Build scored region graph with contact area per edge.

    Accepts float32 or uint8 affinities, uint64 or uint32 segmentation.

    Returns list of dicts ``[{"u": id1, "v": id2, "score": float, "contact_area": int}, ...]``.
    """
    affs = np.ascontiguousarray(affs)
    aff_dtype = affs.dtype
    if aff_dtype == np.float64:
        affs = affs.astype(np.float32)
        aff_dtype = np.dtype("float32")
    if aff_dtype not in _AFF_DTYPE_MAP:
        raise TypeError(f"affs.dtype must be float32 or uint8, got {aff_dtype}")

    seg_dtype = np.dtype(seg_dtype)
    fragments = np.ascontiguousarray(fragments, dtype=seg_dtype)

    module = _compile_module(
        scoring_function, discretize_queue,
        aff_dtype=aff_dtype, seg_dtype=seg_dtype, force_rebuild=force_rebuild,
    )

    return module.buildRegionGraphRich(affs, fragments)
