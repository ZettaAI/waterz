#include <memory>

#include <iostream>
#include <algorithm>
#include <vector>

#include "frontend_agglomerate.h"
#include "evaluate.hpp"
#include "backend/MergeFunctions.hpp"
#include "backend/basic_watershed.hpp"
#include "backend/region_graph.hpp"

#include "backend/SemanticConstraintProvider.hpp"
#include "backend/SemanticTaintConstraintProvider.hpp"
#include "backend/SegConstraintProvider.hpp"
#include "backend/SizeHeuristicConstraintProvider.hpp"
#include "backend/abiss_watershed.hpp"
#include "backend/ShellSummaries.hpp"
#include <type_traits>
#include "backend/wz_threads.hpp"

std::map<int, WaterzContext*> WaterzContext::_contexts;
int WaterzContext::_nextId = 0;

std::vector<ScoredEdge> getRegionGraph(WaterzState& state);
std::vector<double> getRegionGraphMeta(WaterzState& state);

WaterzState
initialize_with_rag(
		const std::vector<ScoredEdge>& rag,
		const std::vector<double>& rag_metadata,
		SegID* segmentation_data,
		std::size_t     width,
		std::size_t     height,
		std::size_t     depth) {

	SegID maxId = 0;
	for (const auto& edge : rag) {
		maxId = std::max(std::max(edge.u, edge.v), maxId);
	}

	std::size_t numNodes = maxId + 1;
	std::cout << "creating region graph for " << numNodes << " nodes" << std::endl;

	std::shared_ptr<RegionGraphType> regionGraph(
			new RegionGraphType(numNodes)
	);

	std::cout << "creating statistics provider" << std::endl;
	std::shared_ptr<StatisticsProviderType> statisticsProvider(
			new StatisticsProviderType(*regionGraph)
	);

	std::cout << "initializing region graph..." << std::endl;

	initialize_with_region_graph<SegID>(
			*statisticsProvider,
			*regionGraph,
			rag,
			rag_metadata);

	std::shared_ptr<ScoringFunctionType> scoringFunction(
			new ScoringFunctionType(*regionGraph, *statisticsProvider)
	);

	std::shared_ptr<RegionMergingType> regionMerging(
			new RegionMergingType(*regionGraph)
	);


	std::shared_ptr<vector<ConstraintProvider*>> dummy_constraints(
		new vector<ConstraintProvider*>()
	);

	WaterzContext* context = WaterzContext::createNew();
	context->regionGraph        = regionGraph;
	context->regionMerging      = regionMerging;
	context->scoringFunction    = scoringFunction;
	context->statisticsProvider = statisticsProvider;
	// context->segmentation       = segmentation_data;
	context->constraints 		= dummy_constraints;

	if (segmentation_data != NULL) {
		// wrap data (no copy)
		volume_ref_ptr<SegID> segmentation(
				new volume_ref<SegID>(
						segmentation_data,
						boost::extents[width][height][depth]
				)
		);
		context->segmentation = segmentation;
	}

	WaterzState initial_state;
	initial_state.context = context->id;

	return initial_state;
}

WaterzState
initialize(
		std::size_t     width,
		std::size_t     height,
		std::size_t     depth,
		const AffValue* affinity_data,
		SegID*          segmentation_data,
		const GtID*     ground_truth_data,
		const SemValue* semantic_data,
		const SegID*    segconstraint_data,
		AffValue        affThresholdLow,
		AffValue        affThresholdHigh,
		AffValue        semantic_aff_threshold,
		size_t          semantic_size_threshold,
		AffValue        semantic_signal_ratio,
		const std::vector<SemValue>& semantic_taint_labels,
		AffValue        semantic_taint_threshold,
		AffValue        size_heuristic_aff_threshold,
		size_t          size_heuristic_small_threshold,
		size_t          size_heuristic_large_threshold,
		bool            findFragments) {

	std::size_t num_voxels = width*height*depth;

	// wrap affinities (no copy)
	affinity_graph_ref<AffValue> affinities(
			affinity_data,
			boost::extents[3][width][height][depth]
	);

	// wrap segmentation array (no copy)
	volume_ref_ptr<SegID> segmentation(
			new volume_ref<SegID>(
					segmentation_data,
					boost::extents[width][height][depth]
			)
	);

	counts_t<std::size_t> sizes;

	if (findFragments) {
		std::cout << "performing initial watershed segmentation..." << std::endl;
		watershed(affinities, affThresholdLow, affThresholdHigh, *segmentation, sizes);
	} else {
		std::cout << "counting regions and sizes..." << std::endl;
		std::size_t maxId = *std::max_element(segmentation_data, segmentation_data + num_voxels);
		sizes.resize(maxId + 1);
		for (std::size_t i = 0; i < num_voxels; i++)
			sizes[segmentation_data[i]]++;
	}

	std::size_t numNodes = sizes.size();
	std::cout << "creating region graph for " << numNodes << " nodes" << std::endl;

	std::shared_ptr<RegionGraphType> regionGraph(
			new RegionGraphType(numNodes)
	);

	std::cout << "creating statistics provider" << std::endl;
	std::shared_ptr<StatisticsProviderType> statisticsProvider(
			new StatisticsProviderType(*regionGraph)
	);

	std::cout << "extracting region graph..." << std::endl;

	get_region_graph(
			affinities,
			*segmentation,
			numNodes - 1,
			*statisticsProvider,
			*regionGraph);

	std::shared_ptr<ScoringFunctionType> scoringFunction(
			new ScoringFunctionType(*regionGraph, *statisticsProvider)
	);

	std::shared_ptr<RegionMergingType> regionMerging(
			new RegionMergingType(*regionGraph)
	);

	std::shared_ptr<vector<ConstraintProvider*>> constraints(
		new vector<ConstraintProvider*>()
	);

	if (semantic_data != NULL) {
		std::cout << "getting semantic information..." << std::endl;
		constraints->push_back(
			new SemanticConstraintProvider<RegionGraphType, SemValue, SegID>(
				semantic_data, segmentation_data, num_voxels,
				semantic_aff_threshold,
				semantic_size_threshold,
				semantic_signal_ratio
			)
		);
	}

	if (semantic_data != NULL && !semantic_taint_labels.empty()) {
		std::cout << "getting semantic taint constraint information..." << std::endl;
		constraints->push_back(
			new SemanticTaintConstraintProvider<RegionGraphType, SemValue, SegID>(
				semantic_data, segmentation_data, num_voxels,
				semantic_taint_labels,
				semantic_taint_threshold
			)
		);
	}

	if (segconstraint_data != NULL) {
		std::cout << "getting seg constraint information..." << std::endl;
		constraints->push_back(
			new SegConstraintProvider<RegionGraphType, SegID>(
				segconstraint_data, segmentation_data, num_voxels
			)
		);
	}

	constraints->push_back(
		new SizeHeuristicConstraintProvider<RegionGraphType, SegID>(
			sizes,
			size_heuristic_aff_threshold,
			size_heuristic_small_threshold,
			size_heuristic_large_threshold
		)
	);

	WaterzContext* context = WaterzContext::createNew();
	context->regionGraph        = regionGraph;
	context->regionMerging      = regionMerging;
	context->scoringFunction    = scoringFunction;
	context->statisticsProvider = statisticsProvider;
	context->segmentation       = segmentation;
	context->constraints 		= constraints;

	WaterzState initial_state;
	initial_state.context = context->id;

	if (ground_truth_data != NULL) {

		// wrap ground-truth (no copy)
		volume_const_ref_ptr<GtID> groundtruth(
				new volume_const_ref<GtID>(
						ground_truth_data,
						boost::extents[width][height][depth]
				)
		);

		context->groundtruth = groundtruth;
	}

	return initial_state;
}


// region_graph_from_codes() with its scan split into z slabs across threads (see the comment at the call).
template <typename Code>
static bool
region_graph_from_codes_slabs(
		std::size_t zdim, std::size_t ydim, std::size_t xdim,
		const uint32_t* seg_data,
		const Code* codes,
		const float* lut,
		StatisticsProviderType& statisticsProvider,
		RegionGraphType& rg) {

	typedef RegionGraphType::EdgeIdType EdgeIdType;
	const int nslabs = std::min<int>(wz_threads(), (int)(zdim / 2));
	if (nslabs <= 1)
		return false;
	const std::size_t slice_size = ydim * xdim;
	const std::size_t aff_channel_size = zdim * slice_size;
	std::vector<std::size_t> z0(nslabs + 1);
	for (int t = 0; t <= nslabs; ++t)
		z0[t] = zdim * t / nslabs;
	// The slabs each fragment's faces can fall in: its own planes' slabs, and for the plane just before a slab,
	// that slab too (its z faces are scanned there).
	std::size_t nfrag = 0;
#pragma omp parallel for num_threads(wz_threads()) schedule(static) reduction(max : nfrag)
	for (std::ptrdiff_t i = 0; i < (std::ptrdiff_t)(zdim * slice_size); ++i)
		nfrag = std::max<std::size_t>(nfrag, seg_data[i]);
	std::vector<uint32_t> reach(nfrag + 1, 0);
	uint32_t* const rch = reach.data();
#pragma omp parallel for num_threads(nslabs) schedule(static, 1)
	for (int t = 0; t < nslabs; ++t) {
		const uint32_t bit = 1u << (t % 32);
		const std::size_t za = z0[t] > 0 ? z0[t] - 1 : 0;
		for (std::size_t i = za * slice_size; i < z0[t + 1] * slice_size;) {
			const uint32_t f = seg_data[i];
			std::size_t j = i + 1;
			while (j < z0[t + 1] * slice_size && seg_data[j] == f)
				++j;
			if (!(__atomic_load_n(&rch[f], __ATOMIC_RELAXED) & bit))
				__atomic_fetch_or(&rch[f], bit, __ATOMIC_RELAXED);
			i = j;
		}
	}
	const bool many_slabs = nslabs > 32;
	struct Slab {
		std::vector<uint64_t> pairs;  // (lo << 32) | hi, in the order the slab first meets them
		std::vector<float> mean;      // the running mean, for an edge with no faces in other slabs
		std::vector<uint32_t> count;
		std::vector<uint8_t> replay;  // the edge's faces are kept and replayed instead
		std::vector<uint32_t> face_pair;  // the kept faces, in scan order
		std::vector<Code> face_code;
	};
	std::vector<Slab> slabs(nslabs);
#pragma omp parallel for num_threads(nslabs) schedule(static, 1)
	for (int t = 0; t < nslabs; ++t) {
		Slab& sl = slabs[t];
		ankerl::unordered_dense::map<uint64_t, uint32_t> index;
		uint64_t lastKey[3] = {UINT64_MAX, UINT64_MAX, UINT64_MAX};
		uint32_t lastPair[3] = {0, 0, 0};
		auto face = [&](int axis, uint32_t id1, uint32_t id2, Code code) {
			const uint32_t lo = std::min(id1, id2), hi = std::max(id1, id2);
			const uint64_t key = ((uint64_t)lo << 32) | (uint64_t)hi;
			uint32_t p;
			if (lastKey[axis] == key) {
				p = lastPair[axis];
			} else {
				auto it = index.try_emplace(key, (uint32_t)sl.pairs.size());
				if (it.second) {
					sl.pairs.push_back(key);
					sl.mean.push_back(0);
					sl.count.push_back(0);
					const uint32_t both = rch[lo] & rch[hi];
					sl.replay.push_back((uint8_t)(many_slabs || (both & (both - 1)) != 0));
				}
				p = it.first->second;
				lastKey[axis] = key;
				lastPair[axis] = p;
			}
			if (sl.replay[p]) {
				sl.face_pair.push_back(p);
				sl.face_code.push_back(code);
			} else {
				// MeanAffinityProvider::addAffinity's arithmetic, on the same values in the same order.
				const AffValue affinity = lo == 0 ? (AffValue)0 : (AffValue)lut[code];
				const std::size_t n = sl.count[p];
				const ScoreValue m = sl.mean[p];
				sl.mean[p] = (affinity + m * n) / (n + 1);
				sl.count[p] = (uint32_t)(n + 1);
			}
		};
		for (std::size_t z = z0[t]; z < z0[t + 1]; ++z)
			for (std::size_t y = 0; y < ydim; ++y)
				for (std::size_t x = 0; x < xdim; ++x) {
					const std::size_t idx = z * slice_size + y * xdim + x;
					const uint32_t id1 = seg_data[idx];
					if (z > 0) {
						const uint32_t id2 = seg_data[idx - slice_size];
						if (id1 != id2)
							face(0, id1, id2, codes[idx]);
					}
					if (y > 0) {
						const uint32_t id2 = seg_data[idx - xdim];
						if (id1 != id2)
							face(1, id1, id2, codes[aff_channel_size + idx]);
					}
					if (x > 0) {
						const uint32_t id2 = seg_data[idx - 1];
						if (id1 != id2)
							face(2, id1, id2, codes[2 * aff_channel_size + idx]);
					}
				}
	}
	std::vector<uint32_t>().swap(reach);
	std::size_t nkept = 0;
	for (const Slab& sl : slabs)
		nkept += sl.face_pair.size();
	std::size_t npairs = 0;
	for (const Slab& sl : slabs)
		npairs += sl.pairs.size();
	rg.reserveEdges(rg.edges().size() + npairs);
	// In slab order: the pairs numbered as the serial scan numbers them, each mean set (an edge with no other
	// slab's faces is new here and complete) or its faces replayed.
	std::vector<EdgeIdType> edgeOf;
	for (Slab& sl : slabs) {
		edgeOf.resize(sl.pairs.size());
		for (std::size_t i = 0; i < sl.pairs.size(); ++i) {
			const SegID lo = (SegID)(sl.pairs[i] >> 32), hi = (SegID)(sl.pairs[i] & 0xFFFFFFFFu);
			// An edge whose fragments meet in one slab only is met here first; only one that can also
			// lie in another slab is looked up.
			EdgeIdType e = sl.replay[i] ? rg.findEdge(lo, hi) : RegionGraphType::NoEdge;
			if (e == RegionGraphType::NoEdge) {
				e = rg.addEdge(lo, hi);
				statisticsProvider.notifyNewEdge(e);
			}
			edgeOf[i] = e;
			if (!sl.replay[i])
				statisticsProvider.addEdge(e, sl.mean[i], (double)sl.count[i]);
		}
		const std::size_t nf = sl.face_pair.size();
		for (std::size_t f = 0; f < nf; ++f) {
			const uint32_t p = sl.face_pair[f];
			const bool zero = (sl.pairs[p] >> 32) == 0;
			const AffValue v = zero ? (AffValue)0 : (AffValue)lut[sl.face_code[f]];
			statisticsProvider.addAffinity(edgeOf[p], v);
		}
		Slab().pairs.swap(sl.pairs);
		std::vector<float>().swap(sl.mean);
		std::vector<uint32_t>().swap(sl.count);
		std::vector<uint8_t>().swap(sl.replay);
		std::vector<uint32_t>().swap(sl.face_pair);
		std::vector<Code>().swap(sl.face_code);
	}
	std::cout << "Region graph number of edges: " << rg.edges().size() << std::endl;
	return true;
}

template <typename Code>
static void
region_graph_from_codes(
		std::size_t zdim, std::size_t ydim, std::size_t xdim,
		const uint32_t* seg_data,
		const Code* codes,
		const float* lut,
		StatisticsProviderType& statisticsProvider,
		RegionGraphType& rg) {

	typedef RegionGraphType::EdgeIdType EdgeIdType;
	const std::size_t slice_size = ydim * xdim;
	const std::size_t aff_channel_size = zdim * slice_size;
	SegID lastA[3] = {0, 0, 0};
	SegID lastB[3] = {0, 0, 0};
	EdgeIdType lastE[3] = {RegionGraphType::NoEdge, RegionGraphType::NoEdge, RegionGraphType::NoEdge};
	auto edgeOf = [&](int axis, SegID id1, SegID id2) -> EdgeIdType {
		if (lastE[axis] != RegionGraphType::NoEdge && lastA[axis] == id1 && lastB[axis] == id2)
			return lastE[axis];
		EdgeIdType e = rg.findEdge(id1, id2);
		if (e == RegionGraphType::NoEdge) {
			e = rg.addEdge(id1, id2);
			statisticsProvider.notifyNewEdge(e);
		}
		lastA[axis] = id1;
		lastB[axis] = id2;
		lastE[axis] = e;
		return e;
	};
	// A face touching fragment 0 carries 0, as the volume path zeroes it before scanning.
	auto value = [&](std::size_t channel, std::size_t idx, SegID id1, SegID id2) -> AffValue {
		if (id1 == 0 || id2 == 0)
			return 0;
		return lut[codes[channel * aff_channel_size + idx]];
	};
	for (std::ptrdiff_t z = 0; z < (std::ptrdiff_t)zdim; ++z)
		for (std::ptrdiff_t y = 0; y < (std::ptrdiff_t)ydim; ++y)
			for (std::ptrdiff_t x = 0; x < (std::ptrdiff_t)xdim; ++x) {
				std::size_t idx = z * slice_size + y * xdim + x;
				SegID id1 = seg_data[idx];
				statisticsProvider.addVoxel(id1, x, y, z);
				if (z > 0) {
					SegID id2 = seg_data[idx - slice_size];
					if (id1 != id2) {
						EdgeIdType e = edgeOf(0, id1, id2);
						statisticsProvider.addAffinity(e, value(0, idx, id1, id2));
					}
				}
				if (y > 0) {
					SegID id2 = seg_data[idx - xdim];
					if (id1 != id2) {
						EdgeIdType e = edgeOf(1, id1, id2);
						statisticsProvider.addAffinity(e, value(1, idx, id1, id2));
					}
				}
				if (x > 0) {
					SegID id2 = seg_data[idx - 1];
					if (id1 != id2) {
						EdgeIdType e = edgeOf(2, id1, id2);
						statisticsProvider.addAffinity(e, value(2, idx, id1, id2));
					}
				}
			}
	std::cout << "Region graph number of edges: " << rg.edges().size() << std::endl;
}

WaterzState
initialize_shell(
		size_t          width,
		size_t          height,
		size_t          depth,
		const uint32_t* fragments,
		size_t          num_nodes,
		const uint8_t*  aff_codes8,
		const uint16_t* aff_codes16,
		const float*    lut,
		const SemValue* semantic_data,
		const SegID*    lowres_constraint,
		size_t lw, size_t lh, size_t ld,
		size_t fw, size_t fh, size_t fd,
		size_t ow, size_t oh, size_t od,
		const SegID*    node_constraint,
		AffValue        semantic_aff_threshold,
		size_t          semantic_size_threshold,
		AffValue        semantic_signal_ratio,
		const std::vector<SemValue>& semantic_taint_labels,
		AffValue        semantic_taint_threshold,
		AffValue        size_heuristic_aff_threshold,
		size_t          size_heuristic_small_threshold,
		size_t          size_heuristic_large_threshold) {

	std::size_t num_voxels = width*height*depth;

	// Sizes, semantic, taint and coarse-constraint summaries in one threaded pass, exactly the serial
	// providers' (see ShellSummaries.hpp); the serial providers below when it declines.
	std::cout << "counting regions and sizes..." << std::endl;
	shell_summaries::Summaries<SemValue, SegID> sums;
	const bool dense = shell_summaries::compute<SemValue, SegID>(
		sums, fragments, width, height, depth, num_nodes, semantic_data, semantic_taint_labels,
		lowres_constraint, lw, lh, ld, fw, fh, fd, ow, oh, od, std::size_t(2) << 30);
	counts_t<std::size_t> sizes;
	if (dense) {
		sizes.swap(sums.sizes);
	} else {
		sizes.resize(num_nodes);
		for (std::size_t i = 0; i < num_voxels; i++)
			sizes[fragments[i]]++;
	}

	std::cout << "creating region graph for " << num_nodes << " nodes" << std::endl;
	std::shared_ptr<RegionGraphType> regionGraph(new RegionGraphType(num_nodes));
	std::shared_ptr<StatisticsProviderType> statisticsProvider(
			new StatisticsProviderType(*regionGraph));

	std::cout << "extracting region graph from codes..." << std::endl;
	// With a provider that keeps nothing per voxel, the scan runs in z slabs across threads and its
	// results are folded in in scan order (region_graph_from_codes_slabs): the same edge numbering,
	// incident lists and running means as the serial scan below.
	bool slabbed = false;
	if (std::is_same<StatisticsProviderType, MeanAffinityProvider<RegionGraphType, ScoreValue>>::value) {
		if (aff_codes8 != NULL)
			slabbed = region_graph_from_codes_slabs<uint8_t>(width, height, depth, fragments, aff_codes8, lut, *statisticsProvider, *regionGraph);
		else
			slabbed = region_graph_from_codes_slabs<uint16_t>(width, height, depth, fragments, aff_codes16, lut, *statisticsProvider, *regionGraph);
	}
	if (slabbed) {
	} else if (aff_codes8 != NULL)
		region_graph_from_codes<uint8_t>(width, height, depth, fragments, aff_codes8, lut, *statisticsProvider, *regionGraph);
	else
		region_graph_from_codes<uint16_t>(width, height, depth, fragments, aff_codes16, lut, *statisticsProvider, *regionGraph);

	std::shared_ptr<ScoringFunctionType> scoringFunction(
			new ScoringFunctionType(*regionGraph, *statisticsProvider));
	std::shared_ptr<RegionMergingType> regionMerging(new RegionMergingType(*regionGraph));
	std::shared_ptr<vector<ConstraintProvider*>> constraints(new vector<ConstraintProvider*>());

	if (dense) {
		if (semantic_data != NULL) {
			std::cout << "getting semantic information..." << std::endl;
			constraints->push_back(
				new DenseSemanticConstraintProvider<RegionGraphType, SemValue>(
					std::move(sums.semantic),
					semantic_aff_threshold, semantic_size_threshold, semantic_signal_ratio));
		}
		if (semantic_data != NULL && !semantic_taint_labels.empty()) {
			std::cout << "getting semantic taint constraint information..." << std::endl;
			constraints->push_back(
				new DenseSemanticTaintConstraintProvider<RegionGraphType>(
					std::move(sums.taint), sizes, semantic_taint_threshold));
		}
		if (lowres_constraint != NULL) {
			std::cout << "getting seg constraint information from the coarse grid..." << std::endl;
			constraints->push_back(
				new DenseSegConstraintProvider<RegionGraphType, SegID>(
					std::move(sums.constraint), std::move(sums.constraint_present)));
		} else if (node_constraint != NULL) {
			std::cout << "getting seg constraint information per node..." << std::endl;
			constraints->push_back(
				new SegConstraintProvider<RegionGraphType, SegID>(node_constraint, num_nodes));
		}
		constraints->push_back(
			new DenseSizeHeuristicConstraintProvider<RegionGraphType, SegID>(
				sizes, size_heuristic_aff_threshold,
				size_heuristic_small_threshold, size_heuristic_large_threshold));
	} else {
	if (semantic_data != NULL) {
		std::cout << "getting semantic information..." << std::endl;
		constraints->push_back(
			new SemanticConstraintProvider<RegionGraphType, SemValue, SegID>(
				semantic_data, fragments, num_voxels,
				semantic_aff_threshold, semantic_size_threshold, semantic_signal_ratio, true));
	}
	if (semantic_data != NULL && !semantic_taint_labels.empty()) {
		std::cout << "getting semantic taint constraint information..." << std::endl;
		constraints->push_back(
			new SemanticTaintConstraintProvider<RegionGraphType, SemValue, SegID>(
				semantic_data, fragments, num_voxels,
				semantic_taint_labels, semantic_taint_threshold, true));
	}
	if (lowres_constraint != NULL) {
		std::cout << "getting seg constraint information from the coarse grid..." << std::endl;
		constraints->push_back(
			new SegConstraintProvider<RegionGraphType, SegID>(
				fragments, width, height, depth,
				lowres_constraint, lw, lh, ld, fw, fh, fd, ow, oh, od));
	} else if (node_constraint != NULL) {
		std::cout << "getting seg constraint information per node..." << std::endl;
		constraints->push_back(
			new SegConstraintProvider<RegionGraphType, SegID>(node_constraint, num_nodes));
	}
	constraints->push_back(
		new SizeHeuristicConstraintProvider<RegionGraphType, SegID>(
			sizes, size_heuristic_aff_threshold,
			size_heuristic_small_threshold, size_heuristic_large_threshold));
	}

	WaterzContext* context = WaterzContext::createNew();
	context->regionGraph        = regionGraph;
	context->regionMerging      = regionMerging;
	context->scoringFunction    = scoringFunction;
	context->statisticsProvider = statisticsProvider;
	context->constraints        = constraints;

	WaterzState initial_state;
	initial_state.context = context->id;
	return initial_state;
}

void
watershedCodes(
		size_t width, size_t height, size_t depth,
		const uint8_t* codes8, const uint16_t* codes16, const float* lut,
		float low, float high, size_t size_threshold, uint32_t* out) {
	if (codes8 != NULL)
		abiss_ws::run<uint8_t>(codes8, lut, width, height, depth, low, high, size_threshold, out);
	else
		abiss_ws::run<uint16_t>(codes16, lut, width, height, depth, low, high, size_threshold, out);
}

std::vector<SegID>
getRoots(WaterzState& state, size_t num_nodes) {
	WaterzContext* context = WaterzContext::get(state.context);
	return context->regionMerging->extractRoots(num_nodes);
}

std::vector<Merge>
mergeUntil(
		WaterzState& state,
		float        threshold) {

	WaterzContext* context = WaterzContext::get(state.context);

	std::cout << "merging until threshold " << (double)threshold << std::endl;

	std::vector<Merge>  mergeHistory;
	MergeHistoryVisitor mergeHistoryVisitor(mergeHistory);

	std::size_t merged = context->regionMerging->mergeUntil(
			*context->scoringFunction,
			*context->statisticsProvider,
			*context->constraints,
			threshold,
			mergeHistoryVisitor
		);

	if (merged && context->segmentation) {

		std::cout << "extracting segmentation" << std::endl;

		context->regionMerging->extractSegmentation(*context->segmentation);
	}

	if (context->groundtruth) {

		std::cout << "evaluating current segmentation against ground-truth" << std::endl;

		auto m = compare_volumes(*context->groundtruth, *context->segmentation);

		state.metrics.rand_split = std::get<0>(m);
		state.metrics.rand_merge = std::get<1>(m);
		state.metrics.voi_split  = std::get<2>(m);
		state.metrics.voi_merge  = std::get<3>(m);
	}

	return mergeHistory;
}

std::vector<ScoredEdge>
getRegionGraph(WaterzState& state) {

	WaterzContext* context = WaterzContext::get(state.context);
	std::shared_ptr<RegionMergingType> regionMerging = context->regionMerging;
	std::shared_ptr<ScoringFunctionType> scoringFunction = context->scoringFunction;

	return regionMerging->extractRegionGraph<ScoredEdge>(*scoringFunction);
}

std::vector<double>
getRegionGraphMeta(WaterzState& state) {

	WaterzContext* context = WaterzContext::get(state.context);
	std::shared_ptr<RegionMergingType> regionMerging = context->regionMerging;
	std::shared_ptr<ScoringFunctionType> scoringFunction = context->scoringFunction;

	return regionMerging->extractRegionGraphMeta<double>(*scoringFunction, *(context->statisticsProvider));
}

std::vector<ScoredEdge>
buildRegionGraphOnly(
		std::size_t     width,
		std::size_t     height,
		std::size_t     depth,
		const AffValue* affinity_data,
		SegID*          segmentation_data) {

	std::size_t num_voxels = width*height*depth;

	// wrap arrays (no copy)
	affinity_graph_ref<AffValue> affinities(
			affinity_data,
			boost::extents[3][width][height][depth]
	);
	volume_ref<SegID> segmentation_vol(
			segmentation_data,
			boost::extents[width][height][depth]
	);

	// count sizes
	std::size_t maxId = *std::max_element(segmentation_data, segmentation_data + num_voxels);
	counts_t<std::size_t> sizes(maxId + 1, 0);
	for (std::size_t i = 0; i < num_voxels; i++)
		sizes[segmentation_data[i]]++;

	std::size_t numNodes = sizes.size();

	// build region graph + statistics (the essential work)
	RegionGraphType regionGraph(numNodes);
	StatisticsProviderType statisticsProvider(regionGraph);

	get_region_graph(
			affinities,
			segmentation_vol,
			numNodes - 1,
			statisticsProvider,
			regionGraph);

	// score edges directly — NO RegionMerging, NO priority queue
	ScoringFunctionType scoringFunction(regionGraph, statisticsProvider);

	std::vector<ScoredEdge> edges;
	edges.reserve(regionGraph.edges().size());

	for (RegionGraphType::EdgeIdType e = 0; e < regionGraph.numEdges(); e++) {
		ScoreValue score = scoringFunction(e);
		edges.push_back(ScoredEdge(
				regionGraph.edge(e).u,
				regionGraph.edge(e).v,
				score));
	}

	return edges;
}

std::vector<RichScoredEdge>
buildRegionGraphRich(
		std::size_t     width,
		std::size_t     height,
		std::size_t     depth,
		const AffValue* affinity_data,
		SegID*          segmentation_data) {

	std::size_t num_voxels = width*height*depth;

	// wrap arrays (no copy)
	affinity_graph_ref<AffValue> affinities(
			affinity_data,
			boost::extents[3][width][height][depth]
	);
	volume_ref<SegID> segmentation_vol(
			segmentation_data,
			boost::extents[width][height][depth]
	);

	// count sizes
	std::size_t maxId = *std::max_element(segmentation_data, segmentation_data + num_voxels);
	counts_t<std::size_t> sizes(maxId + 1, 0);
	for (std::size_t i = 0; i < num_voxels; i++)
		sizes[segmentation_data[i]]++;

	std::size_t numNodes = sizes.size();

	// build region graph + statistics with edge counts
	RegionGraphType regionGraph(numNodes);
	StatisticsProviderType statisticsProvider(regionGraph);
	std::vector<uint64_t> edge_counts;

	get_region_graph(
			affinities,
			segmentation_vol,
			numNodes - 1,
			statisticsProvider,
			regionGraph,
			edge_counts);

	// score edges directly — NO RegionMerging, NO priority queue
	ScoringFunctionType scoringFunction(regionGraph, statisticsProvider);

	std::vector<RichScoredEdge> edges;
	edges.reserve(regionGraph.edges().size());

	for (RegionGraphType::EdgeIdType e = 0; e < regionGraph.numEdges(); e++) {
		ScoreValue score = scoringFunction(e);
		edges.push_back(RichScoredEdge(
				regionGraph.edge(e).u,
				regionGraph.edge(e).v,
				score,
				edge_counts[e]));
	}

	return edges;
}

void
free(WaterzState& state) {

	WaterzContext::free(state.context);
}

void
setNumThreads(int n) {
	wz_thread_setting() = n > 0 ? n : 0;
}

int
getNumThreads() {
	return wz_threads();
}
