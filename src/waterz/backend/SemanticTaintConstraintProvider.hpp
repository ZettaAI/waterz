#ifndef WATERZ_SEMANTIC_TAINT_CONSTRAINT_PROVIDER_H__
#define WATERZ_SEMANTIC_TAINT_CONSTRAINT_PROVIDER_H__

#include "ConstraintProvider.hpp"
#include "DefaultDict.hpp"

#include <unordered_set>
#include <vector>

template <typename RegionGraphType, typename SemValue, typename SegType>
class SemanticTaintConstraintProvider : public ConstraintProvider {

    typedef typename RegionGraphType::NodeIdType NodeIdType;

private:
    DefaultDict<SegType, size_t> _taint_counts{0};
    DefaultDict<SegType, size_t> _total_counts{0};
    float _threshold;

public:
    SemanticTaintConstraintProvider(
        const SemValue* semantic_data,
        const SegType* seg_data,
        size_t num_voxels,
        const std::vector<SemValue>& taint_labels,
        float threshold
    ) : _threshold(threshold) {
        std::unordered_set<SemValue> taint_set(taint_labels.begin(), taint_labels.end());
        // Per run of equal (segment, label) voxels: the same counts with one lookup a run.
        size_t i = 0;
        while (i < num_voxels) {
            const SegType s = seg_data[i];
            const SemValue v = semantic_data[i];
            size_t j = i + 1;
            while (j < num_voxels && seg_data[j] == s && semantic_data[j] == v)
                j++;
            _total_counts[s] += j - i;
            if (taint_set.count(v)) {
                _taint_counts[s] += j - i;
            }
            i = j;
        }
    }

    // The same counts over fragments of another integer type (compact uint32 ids).
    template <typename InSeg>
    SemanticTaintConstraintProvider(
        const SemValue* semantic_data,
        const InSeg* seg_data,
        size_t num_voxels,
        const std::vector<SemValue>& taint_labels,
        float threshold,
        bool /* compact */
    ) : _threshold(threshold) {
        bool taint[256] = {false};
        for (auto l : taint_labels)
            taint[(uint8_t)l] = true;
        size_t i = 0;
        while (i < num_voxels) {
            const InSeg s = seg_data[i];
            const SemValue v = semantic_data[i];
            size_t j = i + 1;
            while (j < num_voxels && seg_data[j] == s && semantic_data[j] == v)
                j++;
            _total_counts[(SegType)s] += j - i;
            if (taint[(uint8_t)v]) {
                _taint_counts[(SegType)s] += j - i;
            }
            i = j;
        }
    }

    inline bool notifyNodeMerge(NodeIdType from, NodeIdType to) override {
        _taint_counts[to] += _taint_counts[from];
        _taint_counts.erase(from);
        _total_counts[to] += _total_counts[from];
        _total_counts.erase(from);
        return true;
    }

    inline bool isConstrained(NodeIdType from, NodeIdType to, float score) const override {
        size_t total_from = _total_counts[from];
        size_t total_to = _total_counts[to];
        bool from_tainted = total_from > 0 && (float)_taint_counts[from] / total_from > _threshold;
        bool to_tainted = total_to > 0 && (float)_taint_counts[to] / total_to > _threshold;
        // tainted segments can only merge with other tainted segments
        if (from_tainted != to_tainted)
            return true;
        return false;
    }
};

#endif // WATERZ_SEMANTIC_TAINT_CONSTRAINT_PROVIDER_H__
