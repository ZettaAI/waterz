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

	std::cout << "counting regions and sizes..." << std::endl;
	counts_t<std::size_t> sizes;
	sizes.resize(num_nodes);
	for (std::size_t i = 0; i < num_voxels; i++)
		sizes[fragments[i]]++;

	std::cout << "creating region graph for " << num_nodes << " nodes" << std::endl;
	std::shared_ptr<RegionGraphType> regionGraph(new RegionGraphType(num_nodes));
	std::shared_ptr<StatisticsProviderType> statisticsProvider(
			new StatisticsProviderType(*regionGraph));

	std::cout << "extracting region graph from codes..." << std::endl;
	if (aff_codes8 != NULL)
		region_graph_from_codes<uint8_t>(width, height, depth, fragments, aff_codes8, lut, *statisticsProvider, *regionGraph);
	else
		region_graph_from_codes<uint16_t>(width, height, depth, fragments, aff_codes16, lut, *statisticsProvider, *regionGraph);

	std::shared_ptr<ScoringFunctionType> scoringFunction(
			new ScoringFunctionType(*regionGraph, *statisticsProvider));
	std::shared_ptr<RegionMergingType> regionMerging(new RegionMergingType(*regionGraph));
	std::shared_ptr<vector<ConstraintProvider*>> constraints(new vector<ConstraintProvider*>());

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
