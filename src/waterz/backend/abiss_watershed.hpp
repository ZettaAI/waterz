#ifndef WATERZ_ABISS_WATERSHED_H__
#define WATERZ_ABISS_WATERSHED_H__

// The abiss watershed (steepest ascent, basin BFS, then single-linkage size merging over the
// max-affinity region graph), ported to read affinity CODES through a dequantization table
// and to keep uint32 labels and a uint32 BFS queue. abiss runs on a float32 copy of the
// affinities padded by one voxel of zeros on every side and channel-last in Fortran order;
// here the padding is virtual (a coordinate on it reads 0) and the codes are read in place,
// so the comparisons see exactly the float values abiss sees and the labels come out the
// same, id for id. Layout: the codes are waterz's (3, Z, Y, X), channels z, y, x; abiss's
// channel c is the x, y, z face, so it reads codes[2 - c]. The padded label volume is indexed
// x + xdim * (y + ydim * z), abiss's Fortran order, which is the same memory order.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <boost/pending/disjoint_sets.hpp>

#include "ankerl/unordered_dense.h"

namespace abiss_ws {

typedef uint32_t Label;
static const Label HIGH_BIT = 0x80000000u;
static const Label ID_MASK = 0x7FFFFFFFu;
static const std::size_t ON_BORDER = (std::size_t(1) << (std::numeric_limits<std::size_t>::digits - 1));

template <typename Code>
struct PaddedAffs {
	const Code* codes;
	const float* lut;
	std::ptrdiff_t X, Y, Z;  // unpadded
	inline float operator()(std::ptrdiff_t x, std::ptrdiff_t y, std::ptrdiff_t z, int c) const {
		if (x <= 0 || y <= 0 || z <= 0 || x > X || y > Y || z > Z)
			return 0.0f;
		const std::size_t idx = ((std::size_t)(z - 1) * Y + (y - 1)) * X + (x - 1);
		return lut[codes[(std::size_t)(2 - c) * X * Y * Z + idx]];
	}
};

// abiss's watershed() with every boundary flag set, as the zetta wrapper calls it.
template <typename Code, typename Index>
inline std::vector<std::size_t>
watershed(const PaddedAffs<Code>& aff, float low, float high, std::vector<Label>& seg_raw) {

	typedef std::ptrdiff_t index;
	const index xdim = aff.X + 2, ydim = aff.Y + 2, zdim = aff.Z + 2;
	const index size = xdim * ydim * zdim;
	seg_raw.assign(size, 0);
	std::vector<std::size_t> counts(1);
	counts[0] = 0;
	auto at = [&](index x, index y, index z) -> Label& { return seg_raw[x + xdim * (y + ydim * z)]; };

	// The border voxels: with every boundary flag set, abiss writes 0 there whatever the
	// affinity (front/back border codes are 0), which the zero-filled volume already holds.
	// Each voxel's flags depend on its own faces alone, so the planes run in parallel.

#pragma omp parallel for schedule(static)
	for (index z = 1; z < zdim - 1; ++z)
		for (index y = 1; y < ydim - 1; ++y)
			for (index x = 1; x < xdim - 1; ++x) {
				Label& id = at(x, y, z) = 0;
				float negx = (x > 1) ? aff(x, y, z, 0) : low;
				float negy = (y > 1) ? aff(x, y, z, 1) : low;
				float negz = (z > 1) ? aff(x, y, z, 2) : low;
				float posx = (x < (xdim - 2)) ? aff(x + 1, y, z, 0) : low;
				float posy = (y < (ydim - 2)) ? aff(x, y + 1, z, 1) : low;
				float posz = (z < (zdim - 2)) ? aff(x, y, z + 1, 2) : low;
				float m = std::max({negx, negy, negz, posx, posy, posz});
				if (m > low) {
					if (negx == m || negx >= high) { id |= 0x01; }
					if (negy == m || negy >= high) { id |= 0x02; }
					if (negz == m || negz >= high) { id |= 0x04; }
					if (posx == m || posx >= high) { id |= 0x08; }
					if (posy == m || posy >= high) { id |= 0x10; }
					if (posz == m || posz >= high) { id |= 0x20; }
				}
			}

	const index dir[6] = {-1, -xdim, -xdim * ydim, 1, xdim, xdim * ydim};
	const Label dirmask[6] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20};
	const Label idirmask[6] = {0x08, 0x10, 0x20, 0x01, 0x02, 0x04};

	Label next_id = 1;
	{
		std::vector<Index> bfs(size + 1, 0);
		index bfs_index = 0;
		index bfs_start = 0;
		index bfs_end = 0;
		for (index iz = 1; iz < zdim - 1; ++iz)
			for (index iy = 1; iy < ydim - 1; ++iy)
				for (index ix = 1; ix < xdim - 1; ++ix) {
					index idx = ix + xdim * (iy + ydim * iz);
					if (!(seg_raw[idx] & HIGH_BIT)) {
						bfs[bfs_end++] = (Index)idx;
						if (seg_raw[idx] == 0) {
							seg_raw[idx] |= HIGH_BIT;
							bfs_start = bfs_end;
							bfs_index = bfs_end;
						} else {
							seg_raw[idx] |= 0x40;
						}
						while (bfs_index != bfs_end) {
							index y = bfs[bfs_index++];
							for (index d = 0; d < 6; ++d) {
								if (seg_raw[y] & dirmask[d]) {
									index z = y + dir[d];
									if (seg_raw[z] & HIGH_BIT) {
										const Label seg_id = seg_raw[z];
										while (bfs_start != bfs_end)
											seg_raw[bfs[bfs_start++]] = seg_id;
										bfs_index = bfs_end;
										d = 6;  // (break)
									} else if (!(seg_raw[z] & 0x40)) {
										seg_raw[z] |= 0x40;
										if (!(seg_raw[z] & idirmask[d])) {  // dfs now
											bfs_index = bfs_end;
											d = 6;  // (break)
										}
										bfs[bfs_end++] = (Index)z;
									}
								}
							}
						}
						if (bfs_start != bfs_end) {
							while (bfs_start != bfs_end)
								seg_raw[bfs[bfs_start++]] = HIGH_BIT | next_id;
							++next_id;
						}
					}
				}
	}
	std::cout << "found: " << (next_id - 1) << " components\n";

	counts.resize(next_id, 0);
	for (index idx = 0; idx < size; ++idx) {
		if (seg_raw[idx] & HIGH_BIT)
			seg_raw[idx] &= ID_MASK;
		else
			seg_raw[idx] = 0;
		++counts[seg_raw[idx]];
	}
	for (index z = 1; z < zdim - 1; ++z)
		for (index y = 1; y < ydim - 1; ++y) {
			--counts[at(0, y, z)];
			--counts[at(xdim - 1, y, z)];
		}
	for (index z = 1; z < zdim - 1; ++z)
		for (index x = 1; x < xdim - 1; ++x) {
			--counts[at(x, 0, z)];
			--counts[at(x, ydim - 1, z)];
		}
	for (index y = 1; y < ydim - 1; ++y)
		for (index x = 1; x < xdim - 1; ++x) {
			--counts[at(x, y, 0)];
			--counts[at(x, y, zdim - 1)];
		}
	return counts;
}

typedef std::vector<std::tuple<float, Label, Label>> RegionGraph;

// The size merge's input: the region graph's edges in abiss's order -- affinity high first,
// ties in the order the pairs were first seen -- cut at the first edge at or below `low`,
// which is where the merge stops reading.
struct SortedEdges {
	std::vector<Label> a, b;
};

// abiss's get_region_graph() with every boundary flag set, as at the pinned commit -- one edge
// per pair of touching labels, its affinity the largest over their faces, the edges in the
// order the pairs are first met by the scan and stable-sorted by affinity, high first -- built
// without a hash map. Labels are numbered in scan order, so the larger label of a face is a
// basin the scan met recently, and usually a small one with few neighbours: each label's
// edges as the larger label are kept on a list threaded through the edge arrays, most
// recently hit first, and a lookup walks a short list that is hot in cache. Affinities are
// carried as their rank among the table's distinct values, so the stable sort is a counting
// sort over the ranks with the same ties. Needs every table value >= 0 (abiss starts each
// edge at 0 and takes the max), which the caller checks.
template <typename Code>
inline SortedEdges
region_graph_sorted(const PaddedAffs<Code>& aff, const std::vector<Label>& seg_raw, std::size_t max_segid,
		float low) {
	typedef std::ptrdiff_t index;
	const index X = aff.X, Y = aff.Y, Z = aff.Z;
	const index xdim = X + 2, ydim = Y + 2;
	const std::size_t ncodes = std::size_t(1) << (8 * sizeof(Code));
	std::vector<float> values(aff.lut, aff.lut + ncodes);
	std::sort(values.begin(), values.end());
	values.erase(std::unique(values.begin(), values.end()), values.end());
	std::vector<uint16_t> rank_of(ncodes);
	for (std::size_t c = 0; c < ncodes; ++c)
		rank_of[c] = (uint16_t)(std::lower_bound(values.begin(), values.end(), aff.lut[c]) - values.begin());
	const std::size_t nranks = values.size();

	// Each label's edges as the larger label are a list threaded through `link` (the other
	// label and the next edge side by side, so a step reads one cache line); the edges
	// themselves are numbered, and their arrays filled, in the order the pairs are first met,
	// which reordering a list does not touch.
	struct Link {
		Label hi;
		uint32_t next;
	};
	std::vector<uint32_t> head(max_segid + 1, UINT32_MAX);
	std::vector<Link> link;
	std::vector<Label> ea;
	std::vector<uint16_t> er;
	const std::size_t guess = std::min<std::size_t>(max_segid * 6, (std::size_t)3 * X * Y * Z);
	link.reserve(guess); ea.reserve(guess); er.reserve(guess);
	auto face = [&](Label s, Label t, uint16_t r) {
		const Label lo = std::min(s, t), hi = std::max(s, t);
		uint32_t prev = UINT32_MAX;
		for (uint32_t e = head[hi]; e != UINT32_MAX; prev = e, e = link[e].next) {
			if (link[e].hi == lo) {
				if (r > er[e]) er[e] = r;
				if (prev != UINT32_MAX) {
					link[prev].next = link[e].next;
					link[e].next = head[hi];
					head[hi] = e;
				}
				return;
			}
		}
		const uint32_t e = (uint32_t)ea.size();
		ea.push_back(hi); er.push_back(r); link.push_back(Link{lo, head[hi]});
		head[hi] = e;
	};
	const std::size_t plane = (std::size_t)X * Y * Z;
	const Code* cz = aff.codes;              // channel z
	const Code* cy = aff.codes + plane;      // channel y
	const Code* cx = aff.codes + 2 * plane;  // channel x
	for (index z = 1; z <= Z; ++z)
		for (index y = 1; y <= Y; ++y) {
			const Label* row = &seg_raw[xdim * (y + ydim * z)];
			const Label* row_y = row - xdim;
			const Label* row_z = row - xdim * ydim;
			const std::size_t base = ((std::size_t)(z - 1) * Y + (y - 1)) * X - 1;
			for (index x = 1; x <= X; ++x) {
				const Label s = row[x];
				if (!s)
					continue;
				if (x > 1) {
					const Label t = row[x - 1];
					if (t && s != t) face(s, t, rank_of[cx[base + x]]);
				}
				if (y > 1) {
					const Label t = row_y[x];
					if (t && s != t) face(s, t, rank_of[cy[base + x]]);
				}
				if (z > 1) {
					const Label t = row_z[x];
					if (t && s != t) face(s, t, rank_of[cz[base + x]]);
				}
			}
		}
	std::vector<uint32_t>().swap(head);
	std::vector<Label> eb(ea.size());
	for (std::size_t e = 0; e < ea.size(); ++e) {
		eb[e] = ea[e];
		ea[e] = link[e].hi;
	}
	std::vector<Link>().swap(link);

	std::vector<std::size_t> start(nranks + 1, 0);
	for (std::size_t e = 0; e < er.size(); ++e)
		++start[nranks - 1 - er[e]];
	std::size_t keep = 0;
	for (std::size_t r = 0; r < nranks; ++r)
		if (values[nranks - 1 - r] > low) keep += start[r];
	std::size_t acc = 0;
	for (std::size_t r = 0; r <= nranks; ++r) {
		const std::size_t n = start[r];
		start[r] = acc;
		acc += n;
	}
	SortedEdges out;
	out.a.resize(keep);
	out.b.resize(keep);
	for (std::size_t e = 0; e < er.size(); ++e) {
		const std::size_t pos = start[nranks - 1 - er[e]]++;
		if (pos < keep) {
			out.a[pos] = ea[e];
			out.b[pos] = eb[e];
		}
	}
	return out;
}

// abiss's get_region_graph() with every boundary flag set, as at the pinned commit: one edge
// map keyed on the packed (min, max) pair, iterated in insertion order -- the first-seen order
// -- then stable-sorted by affinity, high first.
template <typename Code>
inline RegionGraph
region_graph_hashed(const PaddedAffs<Code>& aff, const std::vector<Label>& seg_raw, std::size_t max_segid) {
	typedef std::ptrdiff_t index;
	const index xdim = aff.X + 2, ydim = aff.Y + 2, zdim = aff.Z + 2;
	auto at = [&](index x, index y, index z) -> Label { return seg_raw[x + xdim * (y + ydim * z)]; };
	auto pack = [](Label a, Label b) -> uint64_t {
		return (static_cast<uint64_t>(std::min(a, b)) << 32) | static_cast<uint64_t>(std::max(a, b));
	};
	ankerl::unordered_dense::map<uint64_t, float> edges;
	edges.reserve(max_segid * 5);
	for (index z = 1; z < zdim - 1; ++z)
		for (index y = 1; y < ydim - 1; ++y)
			for (index x = 1; x < xdim - 1; ++x) {
				const Label s = at(x, y, z);
				if (!s)
					continue;
				if (x > 1) {
					const Label t = at(x - 1, y, z);
					if (t && s != t) {
						float& curr = edges[pack(s, t)];
						curr = std::max(curr, aff(x, y, z, 0));
					}
				}
				if (y > 1) {
					const Label t = at(x, y - 1, z);
					if (t && s != t) {
						float& curr = edges[pack(s, t)];
						curr = std::max(curr, aff(x, y, z, 1));
					}
				}
				if (z > 1) {
					const Label t = at(x, y, z - 1);
					if (t && s != t) {
						float& curr = edges[pack(s, t)];
						curr = std::max(curr, aff(x, y, z, 2));
					}
				}
			}
	RegionGraph rg;
	rg.reserve(edges.size());
	for (const auto& kv : edges)
		rg.emplace_back(kv.second, static_cast<Label>(kv.first >> 32), static_cast<Label>(kv.first & 0xFFFFFFFFu));
	std::stable_sort(rg.begin(), rg.end(), [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });
	return rg;
}

template <typename Sets>
inline bool try_merge(std::vector<std::size_t>& counts, Sets& sets, Label s1, Label s2, std::size_t size_threshold) {
	auto real_size_s1 = counts[s1] & (~ON_BORDER);
	auto real_size_s2 = counts[s2] & (~ON_BORDER);
	if ((real_size_s1 >= size_threshold) && (real_size_s2 >= size_threshold))
		return false;
	if (((ON_BORDER & (counts[s1] | counts[s2])) == 0)
			|| (real_size_s1 >= size_threshold && counts[s2] < size_threshold)
			|| (real_size_s2 >= size_threshold && counts[s1] < size_threshold)) {
		counts[s1] += counts[s2] & (~ON_BORDER);
		counts[s1] |= counts[s2] & ON_BORDER;
		counts[s2] = 0;
		sets.link(s1, s2);
		Label s = sets.find_set(s1);
		std::swap(counts[s], counts[s1]);
		return true;
	} else {
		counts[s1] |= counts[s2] & ON_BORDER;
		counts[s2] |= counts[s1] & ON_BORDER;
		return false;
	}
}

// abiss's merge_segments() as the zetta wrapper calls it -- (size, low) thresholds and the
// dust floor both at the size threshold -- for the labels only (its rebuilt region graph is
// output nobody reads here).
inline void merge_segments(std::vector<Label>& seg_raw, const SortedEdges& rg, std::vector<std::size_t>& counts,
		std::size_t size_threshold) {
	std::vector<Label> rank(counts.size());
	std::vector<Label> parent(counts.size());
	boost::disjoint_sets<Label*, Label*> sets(&rank[0], &parent[0]);
	for (Label i = 0; i < counts.size(); i++)
		sets.make_set(i);
	const std::size_t ne = rg.a.size();
	for (std::size_t e = 0; e < ne; ++e) {
		if (e + 16 < ne) {
			__builtin_prefetch(&parent[rg.a[e + 16]]);
			__builtin_prefetch(&parent[rg.b[e + 16]]);
		}
		Label s1 = sets.find_set(rg.a[e]);
		Label s2 = sets.find_set(rg.b[e]);
		if (s1 != s2 && s1 && s2)
			try_merge(counts, sets, s1, s2, size_threshold);
	}
	std::vector<Label> remaps(counts.size());
	counts[0] &= ~ON_BORDER;
	remaps[0] = 0;
	Label next_id = 1;
	for (Label id = 0; id < counts.size(); ++id) {
		Label s = sets.find_set(id);
		if (counts[id] & (~ON_BORDER)) {
			if (s && (counts[s] >= size_threshold)) {
				if (remaps[s] == 0) {
					remaps[s] = next_id;
					counts[next_id] = counts[s] & (~ON_BORDER);
					++next_id;
				}
			} else {
				counts[s] = remaps[s] = 0;
			}
		}
	}
	counts.resize(next_id);
	for (std::size_t idx = 0; idx < seg_raw.size(); ++idx)
		seg_raw[idx] = remaps[sets.find_set(seg_raw[idx])];
	std::cout << "Done with remapping, total: " << (next_id - 1) << std::endl;
}

// The whole of abiss.watershed(): labels for the unpadded volume, C-contiguous (Z, Y, X).
template <typename Code>
inline void run(const Code* codes, const float* lut, std::size_t Z, std::size_t Y, std::size_t X,
		float low, float high, std::size_t size_threshold, uint32_t* out) {
	PaddedAffs<Code> aff{codes, lut, (std::ptrdiff_t)X, (std::ptrdiff_t)Y, (std::ptrdiff_t)Z};
	std::vector<Label> seg_raw;
	const std::size_t padded = (X + 2) * (Y + 2) * (Z + 2);
	std::vector<std::size_t> counts = padded + 1 < (std::size_t(1) << 32)
		? watershed<Code, uint32_t>(aff, low, high, seg_raw)
		: watershed<Code, uint64_t>(aff, low, high, seg_raw);
	if (size_threshold > 0) {
		// The list-threaded graph numbers its edges in 32 bits and ranks the table's values,
		// which reproduces abiss's max-from-0 only when no value is negative; otherwise the
		// hashed graph, sorted the way abiss sorts it.
		const std::size_t ncodes = std::size_t(1) << (8 * sizeof(Code));
		const bool nonnegative = std::all_of(lut, lut + ncodes, [](float v) { return v >= 0.0f; });
		SortedEdges rg;
		if (nonnegative && (std::size_t)3 * X * Y * Z < (std::size_t)UINT32_MAX) {
			rg = region_graph_sorted(aff, seg_raw, counts.size() - 1, low);
		} else {
			RegionGraph hashed = region_graph_hashed(aff, seg_raw, counts.size() - 1);
			for (const auto& e : hashed) {
				if (!(std::get<0>(e) > low))
					break;
				rg.a.push_back(std::get<1>(e));
				rg.b.push_back(std::get<2>(e));
			}
		}
		merge_segments(seg_raw, rg, counts, size_threshold);
	}
	const std::size_t xdim = X + 2, ydim = Y + 2;
	for (std::size_t z = 0; z < Z; ++z)
		for (std::size_t y = 0; y < Y; ++y) {
			const Label* row = &seg_raw[1 + xdim * ((y + 1) + ydim * (z + 1))];
			std::copy(row, row + X, out + (z * Y + y) * X);
		}
}

}  // namespace abiss_ws

#endif  // WATERZ_ABISS_WATERSHED_H__
