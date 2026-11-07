# cython: language_level=3
from libcpp.vector cimport vector
from libc.stdint cimport uint64_t, uint32_t, uint16_t, uint8_t
from libcpp cimport bool
from libcpp.string cimport string

import numpy as np
cimport numpy as np

def agglomerate_rag(
        rag,
        rag_metadata,
        thresholds,
        fragments,
        ):

    if fragments is not None and not fragments.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous fragments arrray (avoid this by passing C_CONTIGUOUS arrays)")
        fragments = np.ascontiguousarray(fragments)

    cdef WaterzState state = __initialize_with_rag(rag, rag_metadata, fragments)

    thresholds.sort()
    for threshold in thresholds:
        merge_history = mergeUntil(state, threshold)
        yield merge_history, fragments

    free(state)


def agglomerate(
        affs: np.array | None,
        thresholds: np.array | None,
        gt: np.array | None,
        fragments: np.array | None,
        semantic: np.array | None,
        segconstraint: np.array | None,

        aff_threshold_low,
        aff_threshold_high,

        semantic_aff_threshold,
        semantic_size_threshold,
        semantic_signal_ratio,

        semantic_taint_labels,
        semantic_taint_threshold,

        size_heuristic_aff_threshold,
        size_heuristic_small_threshold,
        size_heuristic_large_threshold,

        return_merge_history,
        return_region_graph,
        return_region_graph_metadata,
    ):

    # the C++ part assumes contiguous memory, make sure we have it (and do
    # nothing, if we do)
    if affs is not None and not affs.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous affinity arrray (avoid this by passing C_CONTIGUOUS arrays)")
        affs = np.ascontiguousarray(affs)
    if gt is not None and not gt.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous ground-truth arrray (avoid this by passing C_CONTIGUOUS arrays)")
        gt = np.ascontiguousarray(gt)
    if fragments is not None and not fragments.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous fragments arrray (avoid this by passing C_CONTIGUOUS arrays)")
        fragments = np.ascontiguousarray(fragments)
    if semantic is not None and not semantic.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous semantic arrray (avoid this by passing C_CONTIGUOUS arrays)")
        semantic = np.ascontiguousarray(semantic)
    if segconstraint is not None and not segconstraint.flags['C_CONTIGUOUS']:
        print("Creating memory-contiguous segconstraint arrray (avoid this by passing C_CONTIGUOUS arrays)")
        segconstraint = np.ascontiguousarray(segconstraint)

    print("Preparing segmentation volume...")

    if fragments is None:
        volume_shape = (affs.shape[1], affs.shape[2], affs.shape[3])
        segmentation = np.zeros(volume_shape, dtype=np.uint64)
        find_fragments = True
    else:
        segmentation = fragments
        find_fragments = False

    cdef WaterzState state = __initialize(
        affs=affs,
        segmentation=segmentation,
        gt=gt,
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
        find_fragments=find_fragments,
        )

    thresholds.sort()
    for threshold in thresholds:

        merge_history = mergeUntil(state, threshold)

        result = (segmentation,)

        if gt is not None:

            stats = {}
            stats['V_Rand_split'] = state.metrics.rand_split
            stats['V_Rand_merge'] = state.metrics.rand_merge
            stats['V_Info_split'] = state.metrics.voi_split
            stats['V_Info_merge'] = state.metrics.voi_merge

            result += (stats,)

        if return_merge_history:

            result += (merge_history,)

        if return_region_graph:

            result += (getRegionGraph(state),)

        if return_region_graph_metadata:

            result += (getRegionGraphMeta(state),)

        if len(result) == 1:
            yield result[0]
        else:
            yield result

    free(state)

def agglomerate_shell(
        fragments,
        num_nodes,
        aff_codes,
        lut,
        thresholds,
        semantic,
        lowres_constraint,
        factors,
        offsets,
        node_constraint,
        semantic_aff_threshold,
        semantic_size_threshold,
        semantic_signal_ratio,
        semantic_taint_labels,
        semantic_taint_threshold,
        size_heuristic_aff_threshold,
        size_heuristic_small_threshold,
        size_heuristic_large_threshold,
        return_merge_history,
        return_region_graph,
        return_region_graph_metadata,
    ):
    """Agglomerate from compact uint32 fragments and affinity CODES read through ``lut``.

    Yields, per threshold, ``(roots,)`` plus the merge history, region graph and its metadata
    as requested: ``roots[n]`` is the node fragment ``n`` belongs to. Node ids are the
    fragment values. Every array must be C-contiguous; ``aff_codes`` is uint8 or uint16
    ``(3, Z, Y, X)`` with channels z, y, x, and ``lut`` has an entry for every code.
    """
    for arr in (fragments, aff_codes, lut, semantic, lowres_constraint, node_constraint):
        if arr is not None and not arr.flags['C_CONTIGUOUS']:
            raise ValueError("agglomerate_shell needs C-contiguous arrays")
    if aff_codes.dtype not in (np.uint8, np.uint16):
        raise TypeError(f"aff_codes must be uint8 or uint16, got {aff_codes.dtype}")
    if lut.shape[0] != (1 << (8 * aff_codes.dtype.itemsize)):
        raise ValueError("lut must have one entry per code")
    if tuple(aff_codes.shape[1:]) != tuple(fragments.shape) or aff_codes.shape[0] != 3:
        raise ValueError("aff_codes must be (3, *fragments.shape)")
    if node_constraint is not None and node_constraint.shape[0] != num_nodes:
        raise ValueError("node_constraint needs one entry per node")
    cdef WaterzState state = __initialize_shell(
        fragments, num_nodes,
        aff_codes if aff_codes.dtype == np.uint8 else None,
        aff_codes if aff_codes.dtype == np.uint16 else None,
        lut, semantic, lowres_constraint, factors, offsets, node_constraint,
        semantic_aff_threshold, semantic_size_threshold, semantic_signal_ratio,
        semantic_taint_labels, semantic_taint_threshold,
        size_heuristic_aff_threshold, size_heuristic_small_threshold,
        size_heuristic_large_threshold)
    for threshold in sorted(thresholds):
        merge_history = mergeUntil(state, threshold)
        result = (np.asarray(getRoots(state, num_nodes), dtype=np.uint64),)
        if return_merge_history:
            result += (merge_history,)
        if return_region_graph:
            result += (getRegionGraph(state),)
        if return_region_graph_metadata:
            result += (getRegionGraphMeta(state),)
        yield result
    free(state)


def __initialize_shell(
        np.ndarray[uint32_t, ndim=3] fragments,
        size_t num_nodes,
        np.ndarray[uint8_t, ndim=4] codes8,
        np.ndarray[uint16_t, ndim=4] codes16,
        np.ndarray[np.float32_t, ndim=1] lut,
        np.ndarray[uint8_t, ndim=3] semantic,
        np.ndarray[uint64_t, ndim=3] lowres_constraint,
        factors,
        offsets,
        np.ndarray[uint64_t, ndim=1] node_constraint,
        semantic_aff_threshold,
        semantic_size_threshold,
        semantic_signal_ratio,
        semantic_taint_labels,
        semantic_taint_threshold,
        size_heuristic_aff_threshold,
        size_heuristic_small_threshold,
        size_heuristic_large_threshold,
    ):
    cdef const uint8_t* c8 = NULL
    cdef const uint16_t* c16 = NULL
    cdef const uint8_t* sem_data = NULL
    cdef const uint64_t* lr_data = NULL
    cdef const uint64_t* node_data = NULL
    cdef vector[uint8_t] taint_labels_vec
    cdef size_t lw = 0, lh = 0, ld = 0
    for label in semantic_taint_labels:
        taint_labels_vec.push_back(<uint8_t>label)
    if codes8 is not None:
        c8 = &codes8[0, 0, 0, 0]
    else:
        c16 = &codes16[0, 0, 0, 0]
    if semantic is not None:
        sem_data = &semantic[0, 0, 0]
    if lowres_constraint is not None:
        lr_data = &lowres_constraint[0, 0, 0]
        lw = lowres_constraint.shape[0]
        lh = lowres_constraint.shape[1]
        ld = lowres_constraint.shape[2]
    if node_constraint is not None:
        node_data = &node_constraint[0]
    fw, fh, fd = factors
    ow, oh, od = offsets
    return initialize_shell(
        fragments.shape[0], fragments.shape[1], fragments.shape[2],
        &fragments[0, 0, 0], num_nodes, c8, c16, &lut[0], sem_data,
        lr_data, lw, lh, ld, fw, fh, fd, ow, oh, od, node_data,
        semantic_aff_threshold, semantic_size_threshold, semantic_signal_ratio,
        taint_labels_vec, semantic_taint_threshold,
        size_heuristic_aff_threshold, size_heuristic_small_threshold,
        size_heuristic_large_threshold)


def __initialize(
        np.ndarray[np.float32_t, ndim=4] affs,
        np.ndarray[uint64_t, ndim=3] segmentation,
        np.ndarray[uint32_t, ndim=3] gt,
        np.ndarray[uint8_t, ndim=3] semantic,
        np.ndarray[uint64_t, ndim=3] segconstraint,
        aff_threshold_low,
        aff_threshold_high,
        semantic_aff_threshold,
        semantic_size_threshold,
        semantic_signal_ratio,
        semantic_taint_labels,
        semantic_taint_threshold,
        size_heuristic_aff_threshold,
        size_heuristic_small_threshold,
        size_heuristic_large_threshold,
        find_fragments,
    ):

    cdef float*    aff_data
    cdef uint64_t* segmentation_data
    cdef uint32_t* gt_data = NULL
    cdef uint8_t* semantic_data = NULL
    cdef uint64_t* segconstraint_data = NULL
    cdef vector[uint8_t] taint_labels_vec
    for label in semantic_taint_labels:
        taint_labels_vec.push_back(<uint8_t>label)

    aff_data = &affs[0,0,0,0]
    segmentation_data = &segmentation[0,0,0]
    if gt is not None:
        gt_data = &gt[0,0,0]
    if semantic is not None:
        semantic_data = &semantic[0,0,0]
    if segconstraint is not None:
        segconstraint_data = &segconstraint[0,0,0]

    return initialize(
        width=affs.shape[1],
        height=affs.shape[2],
        depth=affs.shape[3],
        affinity_data=aff_data,
        segmentation_data=segmentation_data,
        groundtruth_data=gt_data,
        semantic_data=semantic_data,
        segconstraint_data=segconstraint_data,

        affThresholdLow=aff_threshold_low,
        affThresholdHigh=aff_threshold_high,

        semantic_aff_threshold=semantic_aff_threshold,
        semantic_size_threshold=semantic_size_threshold,
        semantic_signal_ratio=semantic_signal_ratio,

        semantic_taint_labels=taint_labels_vec,
        semantic_taint_threshold=semantic_taint_threshold,

        size_heuristic_aff_threshold=size_heuristic_aff_threshold,
        size_heuristic_small_threshold=size_heuristic_small_threshold,
        size_heuristic_large_threshold=size_heuristic_large_threshold,

        findFragments=find_fragments,
    )

def __initialize_with_rag(
        rag,
        rag_metadata,
        np.ndarray[uint64_t, ndim=3]     segmentation = None):

    cdef uint64_t* segmentation_data = NULL
    shape = (0, 0, 0)

    if segmentation is not None:
        segmentation_data = &segmentation[0,0,0]
        shape = (segmentation.shape[0], segmentation.shape[1], segmentation.shape[2])


    return initialize_with_rag(
        rag,
        rag_metadata,
        segmentation_data,
        shape[0], shape[1], shape[2])

cdef extern from "frontend_agglomerate.h":

    struct Metrics:
        double voi_split
        double voi_merge
        double rand_split
        double rand_merge

    struct Merge:
        uint64_t a
        uint64_t b
        uint64_t c
        float score

    struct ScoredEdge:
        uint64_t u
        uint64_t v
        float score

    struct RichScoredEdge:
        uint64_t u
        uint64_t v
        float score
        uint64_t contact_area

    struct WaterzState:
        int     context
        Metrics metrics

    WaterzState initialize(
            size_t          width,
            size_t          height,
            size_t          depth,
            const float*    affinity_data,
            uint64_t*       segmentation_data,
            const uint32_t* groundtruth_data,
            const uint8_t*  semantic_data,
            const uint64_t* segconstraint_data,
            float           affThresholdLow,
            float           affThresholdHigh,
            float           semantic_aff_threshold,
            uint64_t        semantic_size_threshold,
            float           semantic_signal_ratio,
            const vector[uint8_t]& semantic_taint_labels,
            float           semantic_taint_threshold,
            float           size_heuristic_aff_threshold,
            uint64_t        size_heuristic_small_threshold,
            uint64_t        size_heuristic_large_threshold,
            bool            findFragments);

    WaterzState initialize_with_rag(
            const vector[ScoredEdge]& rag,
            const vector[double]& rag_metadata,
            uint64_t*       segmentation_data,
            size_t          width,
            size_t          height,
            size_t          depth,);

    WaterzState initialize_shell(
            size_t          width,
            size_t          height,
            size_t          depth,
            const uint32_t* fragments,
            size_t          num_nodes,
            const uint8_t*  aff_codes8,
            const uint16_t* aff_codes16,
            const float*    lut,
            const uint8_t*  semantic_data,
            const uint64_t* lowres_constraint,
            size_t lw, size_t lh, size_t ld,
            size_t fw, size_t fh, size_t fd,
            size_t ow, size_t oh, size_t od,
            const uint64_t* node_constraint,
            float           semantic_aff_threshold,
            uint64_t        semantic_size_threshold,
            float           semantic_signal_ratio,
            const vector[uint8_t]& semantic_taint_labels,
            float           semantic_taint_threshold,
            float           size_heuristic_aff_threshold,
            uint64_t        size_heuristic_small_threshold,
            uint64_t        size_heuristic_large_threshold)
    vector[uint64_t] getRoots(WaterzState& state, size_t num_nodes)
    vector[Merge] mergeUntil(
            WaterzState& state,
            float        threshold)

    vector[ScoredEdge] getRegionGraph(WaterzState& state)

    vector[double] getRegionGraphMeta(WaterzState& state)

    vector[ScoredEdge] c_buildRegionGraphOnly "buildRegionGraphOnly" (
            size_t          width,
            size_t          height,
            size_t          depth,
            const float*    affinity_data,
            uint64_t*       segmentation_data)

    vector[RichScoredEdge] c_buildRegionGraphRich "buildRegionGraphRich" (
            size_t          width,
            size_t          height,
            size_t          depth,
            const float*    affinity_data,
            uint64_t*       segmentation_data)

    void free(WaterzState& state)


def buildRegionGraphOnly(
        np.ndarray[np.float32_t, ndim=4] affs,
        np.ndarray[uint64_t, ndim=3] seg):
    """Build scored region graph without RegionMerging overhead."""
    if not affs.flags['C_CONTIGUOUS']:
        affs = np.ascontiguousarray(affs)
    if not seg.flags['C_CONTIGUOUS']:
        seg = np.ascontiguousarray(seg)

    cdef size_t width = affs.shape[1]
    cdef size_t height = affs.shape[2]
    cdef size_t depth = affs.shape[3]
    cdef float* aff_data = &affs[0,0,0,0]
    cdef uint64_t* seg_data = &seg[0,0,0]

    cdef vector[ScoredEdge] edges = c_buildRegionGraphOnly(
        width, height, depth, aff_data, seg_data)

    result = []
    cdef size_t i
    for i in range(edges.size()):
        result.append({
            'u': edges[i].u,
            'v': edges[i].v,
            'score': edges[i].score,
        })
    return result


def buildRegionGraphRich(
        np.ndarray[np.float32_t, ndim=4] affs,
        np.ndarray[uint64_t, ndim=3] seg):
    """Build scored region graph with contact area per edge."""
    if not affs.flags['C_CONTIGUOUS']:
        affs = np.ascontiguousarray(affs)
    if not seg.flags['C_CONTIGUOUS']:
        seg = np.ascontiguousarray(seg)

    cdef size_t width = affs.shape[1]
    cdef size_t height = affs.shape[2]
    cdef size_t depth = affs.shape[3]
    cdef float* aff_data = &affs[0,0,0,0]
    cdef uint64_t* seg_data = &seg[0,0,0]

    cdef vector[RichScoredEdge] edges = c_buildRegionGraphRich(
        width, height, depth, aff_data, seg_data)

    result = []
    cdef size_t i
    for i in range(edges.size()):
        result.append({
            'u': edges[i].u,
            'v': edges[i].v,
            'score': edges[i].score,
            'contact_area': edges[i].contact_area,
        })
    return result
