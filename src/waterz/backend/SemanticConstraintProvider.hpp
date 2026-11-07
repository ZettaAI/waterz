#include "ConstraintProvider.hpp"
#include "DefaultDict.hpp"

#include <cstdint>
#include <unordered_map>

template <typename RegionGraphType, typename SemValue, typename SegType>
class SemanticConstraintProvider : public ConstraintProvider {

	typedef SemValue ValueType;
	typedef typename RegionGraphType::EdgeIdType EdgeIdType;
    typedef typename RegionGraphType::NodeIdType NodeIdType;

private:
    DefaultDict<SegType, DefaultDict<SemValue, size_t>> _semantic{
        DefaultDict<SemValue, size_t>{0}
    };
    float semantic_aff_threshold;
    uint64_t semantic_size_threshold;
    float semantic_signal_ratio;

public:
	SemanticConstraintProvider(
        const SemValue *semantic_data,
        const SegType *seg_data,
        size_t num_voxels,
        float semantic_aff_threshold,
        uint64_t semantic_size_threshold,
        float semantic_signal_ratio
    ) :
    semantic_aff_threshold(semantic_aff_threshold),
    semantic_size_threshold(semantic_size_threshold),
    semantic_signal_ratio(semantic_signal_ratio)
    {
		// One update per RUN of equal (segment, label) voxels rather than per voxel. A run
		// is flushed when it ends, and runs of one segment never overlap, so every label is
		// still inserted into its segment's map in first-seen order: the maps, their
		// iteration order and so every getDictMaxKey tie come out the same.
		std::size_t i = 0;
		while (i < num_voxels) {
			const SegType s = seg_data[i];
			const SemValue v = semantic_data[i];
			std::size_t j = i + 1;
			while (j < num_voxels && seg_data[j] == s && semantic_data[j] == v)
				j++;
			_semantic[s][v] += j - i;
			i = j;
		}
    }

	// The same counts over fragments of another integer type (compact uint32 ids).
	template <typename InSeg>
	SemanticConstraintProvider(
        const SemValue *semantic_data,
        const InSeg *seg_data,
        size_t num_voxels,
        float semantic_aff_threshold,
        uint64_t semantic_size_threshold,
        float semantic_signal_ratio,
        bool /* compact */
    ) :
    semantic_aff_threshold(semantic_aff_threshold),
    semantic_size_threshold(semantic_size_threshold),
    semantic_signal_ratio(semantic_signal_ratio)
    {
		std::size_t i = 0;
		while (i < num_voxels) {
			const InSeg s = seg_data[i];
			const SemValue v = semantic_data[i];
			std::size_t j = i + 1;
			while (j < num_voxels && seg_data[j] == s && semantic_data[j] == v)
				j++;
			_semantic[(SegType)s][v] += j - i;
			i = j;
		}
    }

    inline bool notifyNodeMerge(NodeIdType from, NodeIdType to) {
        for (auto k : _semantic[from].getContainer()) {
            // _semantic[to][k] += _semantic[from][k];
            _semantic[to][k.first] += k.second;
        }
        _semantic.erase(from);
        return true;
    }

    inline bool isConstrained(NodeIdType from, NodeIdType to, float score) const {

        if (score < semantic_aff_threshold)
            return false;

        auto max_sem1_label = getDictMaxKey(_semantic[from].getContainer());
        auto max_sem1 = _semantic[from][max_sem1_label];
        auto total_sem1 = getDictSum(_semantic[from].getContainer());
        auto max_sem2_label = getDictMaxKey(_semantic[to].getContainer());
        auto max_sem2 = _semantic[to][max_sem2_label];
        auto total_sem2 = getDictSum(_semantic[to].getContainer());

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
