#ifndef WATERZ_SHELL_SUMMARIES_H__
#define WATERZ_SHELL_SUMMARIES_H__

// The shell path's per-node summaries -- each node's size, its semantic label counts, its taint
// count and its majority constraint id -- taken in one pass over the fragments with the planes
// split across threads, and the constraint providers that hold them indexed by node rather than
// hashed.
//
// EXACT, not close. The summaries are integer counts, and a count is a sum, so threads adding
// runs into it in any order give the serial count. What else the serial providers decide follows
// from ORDER: each node's semantic labels and constraint ids are inserted into an unordered_map
// in the order the scan first meets them, and the map's iteration order -- which breaks every
// getDictMaxKey tie -- follows from that insertion sequence alone. So the pass also keeps, per
// (node, label) and per (node, constraint id), the flat index of the first voxel, and the maps are
// built by inserting in that order: the same keys in the same order, the same map. The dense
// providers make the same decisions on the same numbers as the hashed ones; a node they have
// erased reads as the hashed ones read an absent key (empty, zero), and an absent constraint
// still throws.

#ifdef _OPENMP
#include <omp.h>
#else
inline int omp_get_max_threads() { return 1; }
inline int omp_get_thread_num() { return 0; }
#endif

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "ConstraintProvider.hpp"
#include "DefaultDict.hpp"
#include "ankerl/unordered_dense.h"
#include "wz_threads.hpp"

namespace shell_summaries {

template <typename Int>
inline void atomic_min(Int* p, Int v) {
	Int cur = __atomic_load_n(p, __ATOMIC_RELAXED);
	while (v < cur && !__atomic_compare_exchange_n(p, &cur, v, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
	}
}

struct NodeCid {
	uint64_t node;
	uint64_t cid;
	bool operator==(const NodeCid& o) const { return node == o.node && cid == o.cid; }
};
struct NodeCidHash {
	using is_avalanching = void;
	uint64_t operator()(const NodeCid& k) const noexcept {
		return ankerl::unordered_dense::detail::wyhash::mix(k.node, k.cid ^ UINT64_C(0x9E3779B97F4A7C15));
	}
};
struct CountFirst {
	uint64_t count;
	uint64_t first;
};

template <typename SemValue, typename SegID>
struct Summaries {
	std::vector<std::size_t> sizes;
	std::vector<DefaultDict<SemValue, std::size_t>> semantic;
	std::vector<std::size_t> taint;
	std::vector<SegID> constraint;
	std::vector<uint8_t> constraint_present;
};

// Everything the serial providers compute, for compact fragments `fragments` (a, b, c) with
// `num_nodes` nodes: semantic (one label per voxel) and the coarse constraint (voxel (a, b, c)
// reads lowres[(a + oa) / fa][(b + ob) / fb][(c + oc) / fc]) are optional. Returns false, having
// done nothing, when the per-(node, label) tables would be larger than `table_budget` bytes; the
// caller then takes the serial providers.
template <typename SemValue, typename SegID>
inline bool compute(Summaries<SemValue, SegID>& out, const uint32_t* fragments, std::size_t na, std::size_t nb,
		std::size_t nc, std::size_t num_nodes, const SemValue* semantic, const std::vector<SemValue>& taint_labels,
		const SegID* lowres, std::size_t la, std::size_t lb, std::size_t lc, std::size_t fa, std::size_t fb,
		std::size_t fc, std::size_t oa, std::size_t ob, std::size_t oc, std::size_t table_budget) {
	(void)la;
	const std::size_t plane = nb * nc;
	const std::size_t nvox = na * plane;
	// The semantic values present, numbered: a node's label counts are a row of that width.
	int lab_of[256];
	std::fill(lab_of, lab_of + 256, -1);
	std::vector<SemValue> values;
	if (semantic != nullptr) {
		bool present[256] = {false};
#pragma omp parallel num_threads(wz_threads())
		{
			bool mine[256] = {false};
#pragma omp for schedule(static)
			for (std::ptrdiff_t a = 0; a < (std::ptrdiff_t)na; ++a) {
				const SemValue* p = semantic + a * plane;
				for (std::size_t i = 0; i < plane; ++i)
					mine[(uint8_t)p[i]] = true;
			}
#pragma omp critical
			for (int v = 0; v < 256; ++v)
				present[v] |= mine[v];
		}
		for (int v = 0; v < 256; ++v)
			if (present[v]) {
				lab_of[v] = (int)values.size();
				values.push_back((SemValue)v);
			}
	}
	const std::size_t L = values.size();
	if (num_nodes * L * (sizeof(uint32_t) + sizeof(uint64_t)) > table_budget)
		return false;

	out.sizes.assign(num_nodes, 0);
	std::vector<uint32_t> lab_count(num_nodes * L, 0);
	std::vector<uint64_t> lab_first(num_nodes * L, UINT64_MAX);
	const int nthreads = wz_threads();
	std::vector<ankerl::unordered_dense::map<NodeCid, CountFirst, NodeCidHash>> cmaps(lowres ? nthreads : 0);
	std::size_t* const sizes = out.sizes.data();
	uint32_t* const lcnt = lab_count.data();
	uint64_t* const lfirst = lab_first.data();

#pragma omp parallel num_threads(wz_threads())
	{
		const int t = omp_get_thread_num();
#pragma omp for schedule(static)
		for (std::ptrdiff_t a = 0; a < (std::ptrdiff_t)na; ++a) {
			const std::size_t ra = (a + oa) / fa;
			for (std::size_t b = 0; b < nb; ++b) {
				const std::size_t row0 = (a * nb + b) * nc;
				const uint32_t* frow = fragments + row0;
				// Runs of one (node, label): its size and its label count, and where it first occurs.
				if (semantic != nullptr) {
					const SemValue* srow = semantic + row0;
					std::size_t c = 0;
					while (c < nc) {
						const uint32_t s = frow[c];
						const SemValue v = srow[c];
						std::size_t c2 = c + 1;
						while (c2 < nc && frow[c2] == s && srow[c2] == v)
							++c2;
						const std::size_t len = c2 - c;
						__atomic_fetch_add(&sizes[s], len, __ATOMIC_RELAXED);
						const std::size_t k = (std::size_t)s * L + lab_of[(uint8_t)v];
						__atomic_fetch_add(&lcnt[k], (uint32_t)len, __ATOMIC_RELAXED);
						if (__atomic_load_n(&lfirst[k], __ATOMIC_RELAXED) > row0 + c)
							atomic_min<uint64_t>(&lfirst[k], row0 + c);
						c = c2;
					}
				} else {
					std::size_t c = 0;
					while (c < nc) {
						const uint32_t s = frow[c];
						std::size_t c2 = c + 1;
						while (c2 < nc && frow[c2] == s)
							++c2;
						__atomic_fetch_add(&sizes[s], c2 - c, __ATOMIC_RELAXED);
						c = c2;
					}
				}
				// Runs of one (node, constraint id), as the serial provider takes them.
				if (lowres != nullptr) {
					auto& m = cmaps[t];
					const SegID* lrow = lowres + (ra * lb + (b + ob) / fb) * lc;
					std::size_t c = 0;
					while (c < nc) {
						const uint32_t s = frow[c];
						const SegID v = lrow[(c + oc) / fc];
						std::size_t c2 = c + 1;
						while (c2 < nc && frow[c2] == s && lrow[(c2 + oc) / fc] == v)
							++c2;
						auto it = m.try_emplace(NodeCid{s, (uint64_t)v}, CountFirst{0, row0 + c}).first;
						it->second.count += c2 - c;
						c = c2;
					}
				}
			}
		}
	}

	// Semantic: each node's labels inserted in the order the scan first met them.
	if (semantic != nullptr) {
		out.semantic.assign(num_nodes, DefaultDict<SemValue, std::size_t>{0});
		out.taint.assign(num_nodes, 0);
		bool is_taint[256] = {false};
		for (auto l : taint_labels)
			is_taint[(uint8_t)l] = true;
		std::vector<std::pair<uint64_t, uint32_t>> order;
		for (std::size_t n = 0; n < num_nodes; ++n) {
			order.clear();
			for (std::size_t j = 0; j < L; ++j)
				if (lcnt[n * L + j])
					order.emplace_back(lfirst[n * L + j], (uint32_t)j);
			if (order.empty())
				continue;
			std::sort(order.begin(), order.end());
			auto& d = out.semantic[n];
			std::size_t taint = 0;
			for (const auto& o : order) {
				const SemValue v = values[o.second];
				const std::size_t cnt = lcnt[n * L + o.second];
				d[v] += cnt;
				if (is_taint[(uint8_t)v])
					taint += cnt;
			}
			out.taint[n] = taint;
		}
	}
	std::vector<uint32_t>().swap(lab_count);
	std::vector<uint64_t>().swap(lab_first);

	// Constraint: per (node, id) the threads' counts summed and the earliest first voxel kept; each
	// node's ids then inserted in first-seen order and the majority read off that map.
	if (lowres != nullptr) {
		struct Entry {
			uint64_t node, cid, count, first;
		};
		std::vector<Entry> all;
		std::size_t total = 0;
		for (const auto& m : cmaps)
			total += m.size();
		all.reserve(total);
		for (auto& m : cmaps) {
			for (const auto& kv : m)
				all.push_back(Entry{kv.first.node, kv.first.cid, kv.second.count, kv.second.first});
			ankerl::unordered_dense::map<NodeCid, CountFirst, NodeCidHash>().swap(m);
		}
		std::sort(all.begin(), all.end(), [](const Entry& x, const Entry& y) {
			return x.node != y.node ? x.node < y.node : (x.cid != y.cid ? x.cid < y.cid : x.first < y.first);
		});
		std::size_t w = 0;
		for (std::size_t i = 0; i < all.size(); ++i) {
			if (w > 0 && all[w - 1].node == all[i].node && all[w - 1].cid == all[i].cid) {
				all[w - 1].count += all[i].count;
				all[w - 1].first = std::min(all[w - 1].first, all[i].first);
			} else {
				all[w++] = all[i];
			}
		}
		all.resize(w);
		std::sort(all.begin(), all.end(), [](const Entry& x, const Entry& y) {
			return x.node != y.node ? x.node < y.node : x.first < y.first;
		});
		out.constraint.assign(num_nodes, 0);
		out.constraint_present.assign(num_nodes, 0);
		std::size_t i = 0;
		while (i < all.size()) {
			std::size_t j = i;
			DefaultDict<SegID, std::size_t> votes{0};
			while (j < all.size() && all[j].node == all[i].node) {
				votes[(SegID)all[j].cid] += all[j].count;
				++j;
			}
			out.constraint[all[i].node] = getDictMaxKey(votes.getContainer());
			out.constraint_present[all[i].node] = 1;
			i = j;
		}
	}
	(void)nvox;
	return true;
}

}  // namespace shell_summaries

template <typename RegionGraphType, typename SemValue>
class DenseSemanticConstraintProvider : public ConstraintProvider {
	typedef typename RegionGraphType::NodeIdType NodeIdType;

	std::vector<DefaultDict<SemValue, std::size_t>> _semantic;
	float semantic_aff_threshold;
	uint64_t semantic_size_threshold;
	float semantic_signal_ratio;

public:
	DenseSemanticConstraintProvider(std::vector<DefaultDict<SemValue, std::size_t>>&& semantic,
			float semantic_aff_threshold, uint64_t semantic_size_threshold, float semantic_signal_ratio)
		: _semantic(std::move(semantic)),
		  semantic_aff_threshold(semantic_aff_threshold),
		  semantic_size_threshold(semantic_size_threshold),
		  semantic_signal_ratio(semantic_signal_ratio) {}

	inline bool notifyNodeMerge(NodeIdType from, NodeIdType to) override {
		for (auto k : _semantic[from].getContainer())
			_semantic[to][k.first] += k.second;
		_semantic[from] = DefaultDict<SemValue, std::size_t>{0};
		return true;
	}

	inline bool isConstrained(NodeIdType from, NodeIdType to, float score) const override {
		if (score < semantic_aff_threshold)
			return false;
		const auto& s1 = _semantic[from];
		const auto& s2 = _semantic[to];
		auto max_sem1_label = getDictMaxKey(s1.getContainer());
		auto max_sem1 = s1[max_sem1_label];
		auto total_sem1 = getDictSum(s1.getContainer());
		auto max_sem2_label = getDictMaxKey(s2.getContainer());
		auto max_sem2 = s2[max_sem2_label];
		auto total_sem2 = getDictSum(s2.getContainer());
		if (total_sem1 < semantic_size_threshold)
			return false;
		if (total_sem2 < semantic_size_threshold)
			return false;
		if (max_sem1 < semantic_signal_ratio * total_sem1)
			return false;
		if (max_sem2 < semantic_signal_ratio * total_sem2)
			return false;
		if (max_sem1_label == 0 || max_sem2_label == 0)
			return false;
		if (max_sem1_label == max_sem2_label)
			return false;
		return true;
	}
};

template <typename RegionGraphType>
class DenseSemanticTaintConstraintProvider : public ConstraintProvider {
	typedef typename RegionGraphType::NodeIdType NodeIdType;

	std::vector<std::size_t> _taint_counts;
	std::vector<std::size_t> _total_counts;
	float _threshold;

public:
	DenseSemanticTaintConstraintProvider(std::vector<std::size_t>&& taint, const std::vector<std::size_t>& sizes,
			float threshold)
		: _taint_counts(std::move(taint)), _total_counts(sizes), _threshold(threshold) {}

	inline bool notifyNodeMerge(NodeIdType from, NodeIdType to) override {
		_taint_counts[to] += _taint_counts[from];
		_taint_counts[from] = 0;
		_total_counts[to] += _total_counts[from];
		_total_counts[from] = 0;
		return true;
	}

	inline bool isConstrained(NodeIdType from, NodeIdType to, float score) const override {
		std::size_t total_from = _total_counts[from];
		std::size_t total_to = _total_counts[to];
		bool from_tainted = total_from > 0 && (float)_taint_counts[from] / total_from > _threshold;
		bool to_tainted = total_to > 0 && (float)_taint_counts[to] / total_to > _threshold;
		if (from_tainted != to_tainted)
			return true;
		return false;
	}
};

template <typename RegionGraphType, typename SegType>
class DenseSegConstraintProvider : public ConstraintProvider {
	typedef typename RegionGraphType::NodeIdType NodeIdType;

	std::vector<SegType> _constraint;
	std::vector<uint8_t> _present;

	inline SegType at(NodeIdType n) const {
		if (!_present[n])
			throw std::out_of_range("DenseSegConstraintProvider: node has no constraint");
		return _constraint[n];
	}

public:
	DenseSegConstraintProvider(std::vector<SegType>&& constraint, std::vector<uint8_t>&& present)
		: _constraint(std::move(constraint)), _present(std::move(present)) {}

	inline bool notifyNodeMerge(NodeIdType from, NodeIdType to) override {
		if (at(to) == 0)
			_constraint[to] = at(from);
		_present[from] = 0;
		return true;
	}

	inline bool isConstrained(NodeIdType from, NodeIdType to, float score) const override {
		if (at(from) == 0 || at(to) == 0)
			return false;
		if (at(from) == at(to))
			return false;
		return true;
	}
};

template <typename RegionGraphType, typename SegType>
class DenseSizeHeuristicConstraintProvider : public ConstraintProvider {
	typedef typename RegionGraphType::NodeIdType NodeIdType;

	std::vector<std::size_t> _size;
	float size_heuristic_aff_threshold;
	std::size_t size_heuristic_small_threshold;
	std::size_t size_heuristic_large_threshold;

public:
	DenseSizeHeuristicConstraintProvider(const std::vector<std::size_t>& sizes, float size_heuristic_aff_threshold,
			std::size_t size_heuristic_small_threshold, std::size_t size_heuristic_large_threshold)
		: _size(sizes),
		  size_heuristic_aff_threshold(size_heuristic_aff_threshold),
		  size_heuristic_small_threshold(size_heuristic_small_threshold),
		  size_heuristic_large_threshold(size_heuristic_large_threshold) {}

	inline bool notifyNodeMerge(NodeIdType from, NodeIdType to) override {
		_size[to] += _size[from];
		_size[from] = 0;
		return true;
	}

	inline bool isConstrained(NodeIdType from, NodeIdType to, float score) const override {
		if (score < size_heuristic_aff_threshold)
			return false;
		if (_size[from] < size_heuristic_small_threshold)
			return false;
		if (_size[to] < size_heuristic_small_threshold)
			return false;
		if ((_size[from] + _size[to]) < size_heuristic_large_threshold)
			return false;
		return true;
	}
};

#endif  // WATERZ_SHELL_SUMMARIES_H__
