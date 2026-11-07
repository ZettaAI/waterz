#ifndef C_FRONTEND_H
#define C_FRONTEND_H

#include <cstdint>
#include <vector>

#include "backend/IterativeRegionMerging.hpp"
#include "backend/MergeFunctions.hpp"
#include "backend/Operators.hpp"
#include "backend/types.hpp"
#include "backend/BinQueue.hpp"
#include "backend/PriorityQueue.hpp"
#include "backend/HistogramQuantileProvider.hpp"
#include "backend/VectorQuantileProvider.hpp"
#include "backend/ConstraintProvider.hpp"
#include "backend/SemanticTaintConstraintProvider.hpp"
#include "evaluate.hpp"

typedef uint32_t GtID;
typedef uint8_t SemValue;

// AffValue, ScoreValue, SegID are defined in the JIT-generated headers below.
// For the pre-compiled default module, these are float/float/uint64_t.
#include <SegType.h>
#include <AffType.h>

typedef RegionGraph<SegID> RegionGraphType;

// to be created by __init__.py
#include <ScoringFunction.h>
#include <Queue.h>

typedef typename ScoringFunctionType::StatisticsProviderType StatisticsProviderType;
typedef IterativeRegionMerging<SegID, ScoreValue, QueueType> RegionMergingType;

struct Metrics {

	double voi_split;
	double voi_merge;
	double rand_split;
	double rand_merge;
};

struct Merge {

	SegID a;
	SegID b;
	SegID c;
	ScoreValue score;
};

struct ScoredEdge {

	ScoredEdge() {};

	ScoredEdge(SegID u_, SegID v_, ScoreValue score_) :
		u(u_),
		v(v_),
		score(score_) {}

	SegID u;
	SegID v;
	ScoreValue score;
};

struct RichScoredEdge {

	RichScoredEdge(SegID u_, SegID v_, ScoreValue score_, uint64_t contact_area_) :
		u(u_),
		v(v_),
		score(score_),
		contact_area(contact_area_) {}

	SegID u;
	SegID v;
	ScoreValue score;
	uint64_t contact_area;
};

struct WaterzState {

	int     context;
	Metrics metrics;
};

class WaterzContext {

public:

	static WaterzContext* createNew() {

		WaterzContext* context = new WaterzContext();
		context->id = _nextId;
		_nextId++;
		_contexts.insert(std::make_pair(context->id, context));

		return context;
	}

	static WaterzContext* get(int id) {

		if (!_contexts.count(id))
			return NULL;

		return _contexts.at(id);
	}

	static void free(int id) {

		WaterzContext* context = get(id);

		if (context) {

			_contexts.erase(id);
			delete context;
		}
	}

	int id;

	std::shared_ptr<RegionGraphType> regionGraph;
	std::shared_ptr<RegionMergingType> regionMerging;
	std::shared_ptr<ScoringFunctionType> scoringFunction;
	std::shared_ptr<StatisticsProviderType> statisticsProvider;
	std::shared_ptr<vector<ConstraintProvider*>> constraints;
	volume_ref_ptr<SegID> segmentation;
	volume_const_ref_ptr<GtID> groundtruth;

private:

	WaterzContext() {}

	~WaterzContext() {}

	static std::map<int, WaterzContext*> _contexts;
	static int _nextId;
};

class RegionMergingVisitor {

public:

	void onPop(RegionGraphType::EdgeIdType e, ScoreValue score) {}

	void onDeletedEdgeFound(RegionGraphType::EdgeIdType e) {}

	void onStaleEdgeFound(RegionGraphType::EdgeIdType e, ScoreValue oldScore, ScoreValue newScore) {}

	void onMerge(SegID a, SegID b, SegID c, ScoreValue score) {}
};

class MergeHistoryVisitor : public RegionMergingVisitor {

public:

	MergeHistoryVisitor(std::vector<Merge>& history) : _history(history) {}

	void onMerge(SegID a, SegID b, SegID c, ScoreValue score) {

		_history.push_back({a, b, c, score});
	}

private:

	std::vector<Merge>& _history;
};

WaterzState initialize(
		size_t          width,
		size_t          height,
		size_t          depth,
		const AffValue* affinity_data,
		SegID*          segmentation_data,
		const GtID*     groundtruth_data,
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
		bool            findFragments
	);

WaterzState initialize_with_rag(
		const std::vector<ScoredEdge>& rag,
		const std::vector<double>& rag_metadata,
		SegID*          segmentation_data,
		std::size_t     width,
		std::size_t     height,
		std::size_t     depth);

// Agglomeration from the SHELL: compact uint32 fragments, affinities as 8- or 16-bit codes
// read through a lookup table, and the constraints as summaries -- no float affinity
// volume, no segmentation volume. Exactly the state initialize() builds from the same
// fragments and the affinities the table produces, with a face touching fragment 0 read as
// 0. Exactly one of aff_codes8 / aff_codes16 is given. The constraint is either a coarse
// grid (lowres_constraint, with integer factors and offsets per axis) or one id per node
// (node_constraint); both NULL means none.
WaterzState initialize_shell(
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
		size_t          size_heuristic_large_threshold);

// abiss's watershed + size merge on affinity codes read through a table, into uint32 labels
// (Z, Y, X); exactly one of codes8 / codes16 is given. See backend/abiss_watershed.hpp.
void watershedCodes(
		size_t width, size_t height, size_t depth,
		const uint8_t* codes8, const uint16_t* codes16, const float* lut,
		float low, float high, size_t size_threshold, uint32_t* out);

// The root of every node after the merges so far.
std::vector<SegID> getRoots(WaterzState& state, size_t num_nodes);

std::vector<Merge> mergeUntil(
		WaterzState& state,
		float        threshold);

std::vector<ScoredEdge> getRegionGraph(WaterzState& state);
std::vector<double> getRegionGraphMeta(WaterzState& state);

std::vector<ScoredEdge> buildRegionGraphOnly(
		std::size_t     width,
		std::size_t     height,
		std::size_t     depth,
		const AffValue* affinity_data,
		SegID*          segmentation_data);

std::vector<RichScoredEdge> buildRegionGraphRich(
		std::size_t     width,
		std::size_t     height,
		std::size_t     depth,
		const AffValue* affinity_data,
		SegID*          segmentation_data);

void free(WaterzState& state);

#endif
