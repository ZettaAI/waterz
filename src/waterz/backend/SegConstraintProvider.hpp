#include "ConstraintProvider.hpp"
#include "DefaultDict.hpp"

#include <unordered_map>

template <typename RegionGraphType, typename SegType>
class SegConstraintProvider : public ConstraintProvider {

    typedef typename RegionGraphType::NodeIdType NodeIdType;

private:
    std::unordered_map<SegType, SegType> _constraint;

public:
	SegConstraintProvider(
            const SegType *segconstraint_data,
            const SegType *seg_data,
            size_t num_voxels) {

        // Collect all mapped constraint, accounting for when an object
        // is straddled across multiple constraint
        DefaultDict<SegType, DefaultDict<SegType, size_t>> constraint{
            DefaultDict<SegType, size_t>{0}
        };
		// Per run of equal (segment, constraint) voxels, flushed when the run ends: each
		// segment's map still receives its keys in first-seen order, so the majority vote
		// and its ties are the same as one update per voxel.
		std::size_t i = 0;
		while (i < num_voxels) {
			const SegType s = seg_data[i];
			const SegType c = segconstraint_data[i];
			std::size_t j = i + 1;
			while (j < num_voxels && seg_data[j] == s && segconstraint_data[j] == c)
				j++;
			constraint[s][c] += j - i;
			i = j;
		}

        // Now we get a single dominant constraint per object
        for (const auto& pair : constraint.getContainer()) {
            const auto& segid = pair.first;
            const auto& mapped_constraint_ids = pair.second;
            _constraint[segid] = getDictMaxKey(mapped_constraint_ids.getContainer());
        }
    }

	// Fragments of another integer type, with the constraint read from a COARSER grid:
	// voxel (a, b, c) takes lowres[(a + oa) / fa][(b + ob) / fb][(c + oc) / fc] -- what a
	// nearest-neighbour upsampling of that grid would hold there, without building it.
	template <typename InSeg>
	SegConstraintProvider(
            const InSeg *seg_data,
            std::size_t na, std::size_t nb, std::size_t nc,
            const SegType *lowres,
            std::size_t la, std::size_t lb, std::size_t lc,
            std::size_t fa, std::size_t fb, std::size_t fc,
            std::size_t oa, std::size_t ob, std::size_t oc) {
        DefaultDict<SegType, DefaultDict<SegType, size_t>> constraint{
            DefaultDict<SegType, size_t>{0}
        };
		std::size_t i = 0;
		for (std::size_t a = 0; a < na; a++) {
			const std::size_t ra = (a + oa) / fa;
			for (std::size_t b = 0; b < nb; b++) {
				const SegType* row = lowres + (ra * lb + (b + ob) / fb) * lc;
				std::size_t c = 0;
				while (c < nc) {
					const InSeg s = seg_data[i];
					const SegType v = row[(c + oc) / fc];
					std::size_t c2 = c + 1;
					while (c2 < nc && seg_data[i + (c2 - c)] == s && row[(c2 + oc) / fc] == v)
						c2++;
					constraint[(SegType)s][v] += c2 - c;
					i += c2 - c;
					c = c2;
				}
			}
		}
        for (const auto& pair : constraint.getContainer())
            _constraint[pair.first] = getDictMaxKey(pair.second.getContainer());
    }

	// One constraint id per node, already decided (a piece-constant constraint volume
	// gives every node exactly its own id as the majority).
	SegConstraintProvider(const SegType *node_constraint, std::size_t num_nodes) {
		for (std::size_t n = 0; n < num_nodes; n++)
			_constraint[(SegType)n] = node_constraint[n];
	}

    inline bool notifyNodeMerge(NodeIdType from, NodeIdType to) {
        if (_constraint.at(to) == 0) {
            _constraint[to] = _constraint.at(from);
        }
        _constraint.erase(from);
        return true;
    }

    inline bool isConstrained(NodeIdType from, NodeIdType to, float score) const {
        if (_constraint.at(from) == 0 || _constraint.at(to) == 0)
            return false;
        if (_constraint.at(from) == _constraint.at(to))
            return false;
        return true;
    }
};
