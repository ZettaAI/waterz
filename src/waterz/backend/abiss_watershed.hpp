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

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <boost/pending/disjoint_sets.hpp>

#include "ankerl/unordered_dense.h"
#include "wz_threads.hpp"

namespace abiss_ws {

typedef uint32_t Label;
static const Label HIGH_BIT = 0x80000000u;
static const Label ID_MASK = 0x7FFFFFFFu;
static const std::size_t ON_BORDER = (std::size_t(1) << (std::numeric_limits<std::size_t>::digits - 1));

// An allocator whose value-initialisation of a trivial element does nothing, so a resize leaves the pages untouched
// until they are written: the padded label volume is written whole by the watershed's first pass, in parallel.
template <typename T>
struct DefaultInitAllocator : std::allocator<T> {
	template <typename U>
	struct rebind {
		using other = DefaultInitAllocator<U>;
	};
	DefaultInitAllocator() = default;
	template <typename U>
	DefaultInitAllocator(const DefaultInitAllocator<U>&) noexcept {}
	template <typename U>
	void construct(U* p) noexcept(std::is_nothrow_default_constructible<U>::value) {
		::new (static_cast<void*>(p)) U;
	}
	template <typename U, typename... Args>
	void construct(U* p, Args&&... args) {
		::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
	}
};
typedef std::vector<Label, DefaultInitAllocator<Label>> LabelVolume;

// Ask for transparent huge pages over the whole pages of [p, p + bytes), before they are first touched: the
// arrays read at random are then one TLB entry per 2 MB rather than per 4 KB. Advisory: same bytes either way.
inline void advise_huge_pages(void* p, std::size_t bytes) {
#ifdef MADV_HUGEPAGE
	const std::size_t page = (std::size_t)sysconf(_SC_PAGESIZE);
	const std::size_t lo = ((std::size_t)p + page - 1) / page * page;
	const std::size_t hi = ((std::size_t)p + bytes) / page * page;
	if (hi > lo)
		madvise(reinterpret_cast<void*>(lo), hi - lo, MADV_HUGEPAGE);
#else
	(void)p;
	(void)bytes;
#endif
}

// A growable array of trivial elements in its own anonymous mapping: it grows by mremap (no copy, no second
// resident copy while it grows), its untouched tail is never resident, and a prefix that has been read for the
// last time can be handed back to the kernel page by page (release_prefix).
template <typename T>
class MappedBuffer {
	T* p_ = nullptr;
	std::size_t n_ = 0, cap_ = 0, released_ = 0;

	static std::size_t bytes(std::size_t n) {
		const std::size_t page = (std::size_t)sysconf(_SC_PAGESIZE);
		return std::max(page, (n * sizeof(T) + page - 1) / page * page);
	}

public:
	MappedBuffer() = default;
	MappedBuffer(const MappedBuffer&) = delete;
	MappedBuffer& operator=(const MappedBuffer&) = delete;
	MappedBuffer(MappedBuffer&& o) noexcept : p_(o.p_), n_(o.n_), cap_(o.cap_), released_(o.released_) {
		o.p_ = nullptr;
		o.n_ = o.cap_ = o.released_ = 0;
	}
	MappedBuffer& operator=(MappedBuffer&& o) noexcept {
		if (this != &o) {
			release();
			p_ = o.p_;
			n_ = o.n_;
			cap_ = o.cap_;
			released_ = o.released_;
			o.p_ = nullptr;
			o.n_ = o.cap_ = o.released_ = 0;
		}
		return *this;
	}
	~MappedBuffer() { release(); }

	void reserve(std::size_t c) {
		if (c <= cap_)
			return;
		void* q = p_ == nullptr
			? mmap(nullptr, bytes(c), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0)
			: mremap(p_, bytes(cap_), bytes(c), MREMAP_MAYMOVE);
		if (q == MAP_FAILED)
			throw std::bad_alloc();
		p_ = static_cast<T*>(q);
		cap_ = c;
	}
	void resize(std::size_t n) {
		reserve(n);
		n_ = n;
	}
	// resize(), the new pages asked for as huge pages before anything touches them.
	void resize_huge(std::size_t n) {
		const bool fresh = p_ == nullptr;
		reserve(n);
		if (fresh)
			advise_huge_pages(p_, bytes(cap_));
		n_ = n;
	}
	inline void push_back(const T& v) {
		if (n_ == cap_)
			reserve(cap_ ? cap_ + cap_ / 2 : 4096);
		p_[n_++] = v;
	}
	// The first n elements are never read again: their whole pages go back to the kernel.
	void release_prefix(std::size_t n) {
		const std::size_t page = (std::size_t)sysconf(_SC_PAGESIZE);
		const std::size_t upto = (n * sizeof(T)) / page * page;
		if (upto > released_) {
			madvise(reinterpret_cast<char*>(p_) + released_, upto - released_, MADV_DONTNEED);
			released_ = upto;
		}
	}
	void release() {
		if (p_ != nullptr)
			munmap(p_, bytes(cap_));
		p_ = nullptr;
		n_ = cap_ = released_ = 0;
	}
	inline T& operator[](std::size_t i) { return p_[i]; }
	inline const T& operator[](std::size_t i) const { return p_[i]; }
	std::size_t size() const { return n_; }
	T* data() { return p_; }
	const T* data() const { return p_; }
};

// A size-merge edge, (smaller label, larger label).
struct EdgePair {
	Label lo, hi;
};

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
watershed(const PaddedAffs<Code>& aff, float low, float high, LabelVolume& seg_raw) {

	typedef std::ptrdiff_t index;
	const index xdim = aff.X + 2, ydim = aff.Y + 2, zdim = aff.Z + 2;
	const index size = xdim * ydim * zdim;
	seg_raw.resize(size);
	advise_huge_pages(seg_raw.data(), size * sizeof(Label));
	std::vector<std::size_t> counts(1);
	counts[0] = 0;
	auto at = [&](index x, index y, index z) -> Label& { return seg_raw[x + xdim * (y + ydim * z)]; };

	// The border voxels: with every boundary flag set, abiss writes 0 there whatever the
	// affinity (front/back border codes are 0); the volume is not zero-filled, so they are
	// written here, and every interior voxel by the flags below. Each voxel's flags depend on
	// its own faces alone, so the planes run in parallel.
	std::fill(seg_raw.begin(), seg_raw.begin() + xdim * ydim, Label(0));
	std::fill(seg_raw.end() - xdim * ydim, seg_raw.end(), Label(0));

#pragma omp parallel for num_threads(wz_threads()) schedule(static)
	for (index z = 1; z < zdim - 1; ++z) {
		std::fill(&at(0, 0, z), &at(0, 0, z) + xdim, Label(0));
		std::fill(&at(0, ydim - 1, z), &at(0, ydim - 1, z) + xdim, Label(0));
		for (index y = 1; y < ydim - 1; ++y) {
			at(0, y, z) = 0;
			at(xdim - 1, y, z) = 0;
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
		}
	}

	const index dir[6] = {-1, -xdim, -xdim * ydim, 1, xdim, xdim * ydim};
	const Label dirmask[6] = {0x01, 0x02, 0x04, 0x08, 0x10, 0x20};
	const Label idirmask[6] = {0x08, 0x10, 0x20, 0x01, 0x02, 0x04};

	Label next_id = 1;
	{
		// Each search starts on an empty queue (every search ends with start == end == index), so it is
		// emptied there and only the longest single search is ever written.
		std::unique_ptr<Index[]> bfs(new Index[size + 1]);
		index bfs_index = 0;
		index bfs_start = 0;
		index bfs_end = 0;
		for (index iz = 1; iz < zdim - 1; ++iz)
			for (index iy = 1; iy < ydim - 1; ++iy)
				for (index ix = 1; ix < xdim - 1; ++ix) {
					index idx = ix + xdim * (iy + ydim * iz);
					if (!(seg_raw[idx] & HIGH_BIT)) {
						bfs_start = bfs_end = bfs_index = 0;
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

	// Strip the flags and count each label, the planes in parallel: a count is a sum, so adding each
	// run of one label once, atomically, gives the serial counts exactly.
	counts.resize(next_id, 0);
	std::size_t* const cnt = counts.data();
#pragma omp parallel for num_threads(wz_threads()) schedule(static)
	for (index z = 0; z < zdim; ++z) {
		Label* const plane = &seg_raw[xdim * ydim * z];
		const index n = xdim * ydim;
		Label run = 0;
		std::size_t len = 0;
		for (index i = 0; i < n; ++i) {
			const Label v = (plane[i] & HIGH_BIT) ? (plane[i] & ID_MASK) : Label(0);
			plane[i] = v;
			if (v != run) {
				if (len) __atomic_fetch_add(&cnt[run], len, __ATOMIC_RELAXED);
				run = v;
				len = 0;
			}
			++len;
		}
		if (len) __atomic_fetch_add(&cnt[run], len, __ATOMIC_RELAXED);
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
region_graph_sorted(const PaddedAffs<Code>& aff, const LabelVolume& seg_raw, std::size_t max_segid,
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


// region_graph_sorted() in 10 bytes an edge: the same faces through the same list-threaded lookup, so the
// same edges are made in the same order with the same ranks; the larger label is not stored per edge but
// recovered from the list each edge is on, and the sorted pairs are written as the edges are read.
template <typename Code>
inline MappedBuffer<EdgePair>
region_graph_lean(const PaddedAffs<Code>& aff, const LabelVolume& seg_raw, std::size_t max_segid, float low) {
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

	// link[e] is (the smaller label, the next edge on the larger label's list) while the graph is built,
	// and (the smaller label, the larger label) once the lists have been read off.
	MappedBuffer<uint32_t> head;
	head.resize_huge(max_segid + 1);
	std::fill(head.data(), head.data() + max_segid + 1, UINT32_MAX);
	MappedBuffer<EdgePair> link;
	MappedBuffer<uint16_t> er;
	const std::size_t guess = std::min<std::size_t>(max_segid * 6, (std::size_t)3 * X * Y * Z);
	link.reserve(guess);
	er.reserve(guess);
	auto face = [&](Label s, Label t, uint16_t r) {
		const Label lo = std::min(s, t), hi = std::max(s, t);
		uint32_t prev = UINT32_MAX;
		for (uint32_t e = head[hi]; e != UINT32_MAX; prev = e, e = link[e].hi) {
			if (link[e].lo == lo) {
				if (r > er[e]) er[e] = r;
				if (prev != UINT32_MAX) {
					link[prev].hi = link[e].hi;
					link[e].hi = head[hi];
					head[hi] = e;
				}
				return;
			}
		}
		const uint32_t e = (uint32_t)link.size();
		er.push_back(r);
		link.push_back(EdgePair{lo, head[hi]});
		head[hi] = e;
	};
	const std::size_t plane = (std::size_t)X * Y * Z;
	const Code* cz = aff.codes;
	const Code* cy = aff.codes + plane;
	const Code* cx = aff.codes + 2 * plane;
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
	// Every edge is on exactly one list, its larger label's; the lists are disjoint, so label ranges go to
	// threads, and each thread walks a batch of lists side by side so their cache misses overlap.
	EdgePair* const lk = link.data();
	const uint32_t* const hd = head.data();
	const std::ptrdiff_t nlab = (std::ptrdiff_t)max_segid + 1;
#pragma omp parallel for num_threads(wz_threads()) schedule(dynamic, 1)
	for (std::ptrdiff_t b0 = 0; b0 < nlab; b0 += 65536) {
		const std::ptrdiff_t b1 = std::min<std::ptrdiff_t>(b0 + 65536, nlab);
		const int W = 16;
		uint32_t cur[W];
		Label lab[W];
		int live = 0;
		std::ptrdiff_t nextLabel = b0;
		auto refill = [&](int i) -> bool {
			while (nextLabel < b1) {
				const uint32_t e = hd[nextLabel];
				const Label l = (Label)nextLabel++;
				if (e != UINT32_MAX) {
					cur[i] = e;
					lab[i] = l;
					return true;
				}
			}
			return false;
		};
		for (int i = 0; i < W; ++i)
			if (refill(live))
				++live;
		while (live > 0) {
			for (int i = 0; i < live;) {
				const uint32_t e = cur[i];
				const uint32_t nxt = lk[e].hi;
				lk[e].hi = lab[i];
				if (nxt != UINT32_MAX) {
					cur[i] = nxt;
					++i;
				} else if (refill(i)) {
					++i;
				} else {
					--live;
					cur[i] = cur[live];
					lab[i] = lab[live];
				}
			}
		}
	}
	head.release();

	std::vector<std::size_t> start(nranks + 1, 0);
	const std::size_t ne = er.size();
	for (std::size_t e = 0; e < ne; ++e)
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
	MappedBuffer<EdgePair> out;
	out.resize(keep);
	const std::size_t STEP = std::size_t(1) << 23;
	for (std::size_t e = 0; e < ne; ++e) {
		const std::size_t pos = start[nranks - 1 - er[e]]++;
		if (pos < keep)
			out[pos] = lk[e];
		if ((e + 1) % STEP == 0) {
			link.release_prefix(e + 1);
			er.release_prefix(e + 1);
		}
	}
	return out;
}


// region_graph_lean() with the scan split into z slabs across threads (see the comment at the call): the same
// edges, the same order, the same ranks, the same sorted pairs. Returns false, having done nothing, when the
// volume is too thin for the slabs.
template <typename Code>
inline bool
region_graph_lean_slabs(const PaddedAffs<Code>& aff, const LabelVolume& seg_raw, std::size_t max_segid, float low,
		MappedBuffer<EdgePair>& result) {
	typedef std::ptrdiff_t index;
	const index X = aff.X, Y = aff.Y, Z = aff.Z;
	const index xdim = X + 2, ydim = Y + 2;
	const int S = std::min<int>(wz_threads(), (int)(Z / 8));
	if (S < 2)
		return false;
	const std::size_t ncodes = std::size_t(1) << (8 * sizeof(Code));
	std::vector<float> values(aff.lut, aff.lut + ncodes);
	std::sort(values.begin(), values.end());
	values.erase(std::unique(values.begin(), values.end()), values.end());
	std::vector<uint16_t> rank_of(ncodes);
	for (std::size_t c = 0; c < ncodes; ++c)
		rank_of[c] = (uint16_t)(std::lower_bound(values.begin(), values.end(), aff.lut[c]) - values.begin());
	const std::size_t nranks = values.size();
	const uint32_t NONE = UINT32_MAX;
	const Label DEAD = 0x80000000u;

	// Slab s scans interior planes [z0[s], z0[s + 1]); its own labels are [first[s], first[s + 1]).
	std::vector<index> z0(S + 1);
	for (int t = 0; t <= S; ++t)
		z0[t] = 1 + (Z * t) / S;
	std::vector<Label> plane_max(Z + 2, 0);
#pragma omp parallel for num_threads(wz_threads()) schedule(static)
	for (index z = 1; z <= Z; ++z) {
		Label m = 0;
		const Label* p = &seg_raw[xdim * ydim * z];
		for (index i = 0; i < xdim * ydim; ++i)
			m = std::max(m, p[i]);
		plane_max[z] = m;
	}
	std::vector<std::size_t> first(S + 1);
	{
		Label m = 0;
		int t = 0;
		for (index z = 1; z <= Z + 1 && t <= S; ++z) {
			while (t <= S && z0[t] == z) {
				first[t] = (std::size_t)m + 1;
				++t;
			}
			if (z <= Z)
				m = std::max(m, plane_max[z]);
		}
		first[S] = max_segid + 1;
	}
	// The last slab each label occurs in, or meets a label of through a face.
	std::vector<uint8_t> last(max_segid + 1, 0);

	struct Slab {
		MappedBuffer<EdgePair> link;  // (lo, next on hi's list) while built, then (lo, hi)
		MappedBuffer<uint16_t> er;
		std::vector<uint32_t> foreign;  // indices of the edges between two earlier slabs' labels
		std::vector<uint32_t> candidates;  // edges whose two labels both reach a later slab
	};
	std::vector<Slab> slabs(S);
	const std::size_t plane = (std::size_t)X * Y * Z;
	const Code* cz = aff.codes;
	const Code* cy = aff.codes + plane;
	const Code* cx = aff.codes + 2 * plane;
	uint8_t* const lastp = last.data();

#pragma omp parallel for num_threads(S) schedule(static, 1)
	for (int t = 0; t < S; ++t) {
		Slab& sl = slabs[t];
		const std::size_t own0 = first[t], own1 = first[t + 1];
		MappedBuffer<uint32_t> head;
		head.resize_huge(own1 > own0 ? own1 - own0 : 1);
		std::fill(head.data(), head.data() + head.size(), NONE);
		ankerl::unordered_dense::map<uint64_t, uint32_t> fmap;
		const std::size_t guess = std::min<std::size_t>((own1 - own0) * 6 + 1024, (std::size_t)3 * X * Y * (z0[t + 1] - z0[t]));
		sl.link.reserve(guess);
		sl.er.reserve(guess);
		const uint8_t me = (uint8_t)t;
		// The last pair each face direction met and its edge: neighbouring voxels along a row mostly face the
		// same pair, which then needs no lookup. A hit skips only the list's move to front, which changes
		// nothing but the order lookups walk a list in.
		struct Last {
			Label lo = 0, hi = 0;
			uint32_t e = NONE;
		};
		Last last_x, last_y, last_z;
		auto face = [&](Label a, Label b, uint16_t r, Last& c) {
			const Label lo = std::min(a, b), hi = std::max(a, b);
			if (c.lo == lo && c.hi == hi) {
				if (r > sl.er[c.e]) sl.er[c.e] = r;
				return;
			}
			c.lo = lo;
			c.hi = hi;
			if (hi >= own0) {
				uint32_t* const h = &head[hi - own0];
				uint32_t prev = NONE;
				for (uint32_t e = *h; e != NONE; prev = e, e = sl.link[e].hi) {
					if (sl.link[e].lo == lo) {
						if (r > sl.er[e]) sl.er[e] = r;
						if (prev != NONE) {
							sl.link[prev].hi = sl.link[e].hi;
							sl.link[e].hi = *h;
							*h = e;
						}
						c.e = e;
						return;
					}
				}
				const uint32_t e = (uint32_t)sl.link.size();
				sl.er.push_back(r);
				sl.link.push_back(EdgePair{lo, *h});
				*h = e;
				c.e = e;
			} else {
				const uint64_t key = ((uint64_t)lo << 32) | hi;
				auto it = fmap.try_emplace(key, (uint32_t)sl.link.size());
				if (it.second) {
					sl.foreign.push_back((uint32_t)sl.link.size());
					sl.er.push_back(r);
					sl.link.push_back(EdgePair{lo, hi});
				} else if (r > sl.er[it.first->second]) {
					sl.er[it.first->second] = r;
				}
				c.e = it.first->second;
			}
		};
		for (index z = z0[t]; z < z0[t + 1]; ++z)
			for (index y = 1; y <= Y; ++y) {
				const Label* row = &seg_raw[xdim * (y + ydim * z)];
				const Label* row_y = row - xdim;
				const Label* row_z = row - xdim * ydim;
				const std::size_t base = ((std::size_t)(z - 1) * Y + (y - 1)) * X - 1;
				Label run = 0;
				// The plane before the slab's first one is the earlier slab's; its labels meet this
				// slab's through the z faces here, so they count as reaching this slab too.
				if (z == z0[t] && z > 1) {
					Label zrun = 0;
					for (index x = 1; x <= X; ++x) {
						const Label u = row_z[x];
						if (u && u != zrun) {
							zrun = u;
							uint8_t cur = __atomic_load_n(&lastp[u], __ATOMIC_RELAXED);
							while (cur < me && !__atomic_compare_exchange_n(&lastp[u], &cur, me, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
							}
						}
					}
				}
				for (index x = 1; x <= X; ++x) {
					const Label s = row[x];
					if (!s)
						continue;
					if (s != run) {
						run = s;
						uint8_t cur = __atomic_load_n(&lastp[s], __ATOMIC_RELAXED);
						while (cur < me && !__atomic_compare_exchange_n(&lastp[s], &cur, me, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
						}
					}
					if (x > 1) {
						const Label u = row[x - 1];
						if (u && s != u) face(s, u, rank_of[cx[base + x]], last_x);
					}
					if (y > 1) {
						const Label u = row_y[x];
						if (u && s != u) face(s, u, rank_of[cy[base + x]], last_y);
					}
					if (z > 1) {
						const Label u = row_z[x];
						if (u && s != u) face(s, u, rank_of[cz[base + x]], last_z);
					}
				}
			}
		// Each own label's list, read off into the larger label of its edges.
		EdgePair* const lk = sl.link.data();
		for (std::size_t h = own0; h < own1; ++h) {
			uint32_t e = head[h - own0];
			while (e != NONE) {
				const uint32_t nxt = lk[e].hi;
				lk[e].hi = (Label)h;
				e = nxt;
			}
		}
	}
#pragma omp parallel for num_threads(S) schedule(static, 1)
	for (int t = 0; t < S; ++t) {
		Slab& sl = slabs[t];
		const EdgePair* const lk = sl.link.data();
		for (std::size_t e = 0; e < sl.link.size(); ++e)
			if (lastp[lk[e].hi] > t && lastp[lk[e].lo] > t)
				sl.candidates.push_back((uint32_t)e);
	}
	// In slab order: an edge met before (only ever between two earlier slabs' labels) is dropped and its rank
	// folded into the first copy; an edge that can still be met later is entered for its later copies.
	{
		ankerl::unordered_dense::map<uint64_t, uint64_t> seen;
		for (int t = 0; t < S; ++t) {
			Slab& sl = slabs[t];
			for (const uint32_t e : sl.foreign) {
				const uint64_t key = ((uint64_t)sl.link[e].lo << 32) | sl.link[e].hi;
				auto it = seen.find(key);
				if (it != seen.end()) {
					Slab& orig = slabs[it->second >> 32];
					const uint32_t oe = (uint32_t)(it->second & 0xFFFFFFFFu);
					if (sl.er[e] > orig.er[oe]) orig.er[oe] = sl.er[e];
					sl.link[e].hi |= DEAD;
				}
			}
			for (const uint32_t e : sl.candidates) {
				if (sl.link[e].hi & DEAD)
					continue;
				const uint64_t key = ((uint64_t)sl.link[e].lo << 32) | sl.link[e].hi;
				seen.emplace(key, ((uint64_t)t << 32) | e);
			}
			std::vector<uint32_t>().swap(sl.foreign);
			std::vector<uint32_t>().swap(sl.candidates);
		}
	}
	std::vector<uint8_t>().swap(last);
	// The counting sort: each rank's range of slots is cut into the slabs' shares in slab order.
	std::vector<std::vector<std::size_t>> hist(S, std::vector<std::size_t>(nranks, 0));
#pragma omp parallel for num_threads(S) schedule(static, 1)
	for (int t = 0; t < S; ++t) {
		const Slab& sl = slabs[t];
		for (std::size_t e = 0; e < sl.link.size(); ++e)
			if (!(sl.link[e].hi & DEAD))
				++hist[t][nranks - 1 - sl.er[e]];
	}
	std::size_t keep = 0;
	for (std::size_t r = 0; r < nranks; ++r)
		if (values[nranks - 1 - r] > low)
			for (int t = 0; t < S; ++t)
				keep += hist[t][r];
	std::vector<std::vector<std::size_t>> start(S, std::vector<std::size_t>(nranks, 0));
	{
		std::size_t acc = 0;
		for (std::size_t r = 0; r < nranks; ++r)
			for (int t = 0; t < S; ++t) {
				start[t][r] = acc;
				acc += hist[t][r];
			}
	}
	result.resize(keep);
	EdgePair* const out = result.data();
	const std::size_t STEP = std::size_t(1) << 23;
#pragma omp parallel for num_threads(S) schedule(static, 1)
	for (int t = 0; t < S; ++t) {
		Slab& sl = slabs[t];
		std::vector<std::size_t>& st = start[t];
		const std::size_t ne = sl.link.size();
		for (std::size_t e = 0; e < ne; ++e) {
			const EdgePair p = sl.link[e];
			if (!(p.hi & DEAD)) {
				const std::size_t pos = st[nranks - 1 - sl.er[e]]++;
				if (pos < keep)
					out[pos] = p;
			}
			if ((e + 1) % STEP == 0) {
				sl.link.release_prefix(e + 1);
				sl.er.release_prefix(e + 1);
			}
		}
		sl.link.release();
		sl.er.release();
	}
	return true;
}

// abiss's get_region_graph() with every boundary flag set, as at the pinned commit: one edge
// map keyed on the packed (min, max) pair, iterated in insertion order -- the first-seen order
// -- then stable-sorted by affinity, high first.
template <typename Code>
inline RegionGraph
region_graph_hashed(const PaddedAffs<Code>& aff, const LabelVolume& seg_raw, std::size_t max_segid) {
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
inline void merge_segments_reference(LabelVolume& seg_raw, const SortedEdges& rg, std::vector<std::size_t>& counts,
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
	// Every label's new id, through its representative, once per label (the sets no longer change, and
	// find_set reads only the parents, so `rank` is free to hold it); the voxels then read a table, in
	// parallel.
	const std::size_t nlabels = remaps.size();
	for (std::size_t id = 0; id < nlabels; ++id)
		rank[id] = remaps[sets.find_set((Label)id)];
	const Label* const final_id = rank.data();
	Label* const vox = seg_raw.data();
	const std::ptrdiff_t nvox = (std::ptrdiff_t)seg_raw.size();
#pragma omp parallel for num_threads(wz_threads()) schedule(static)
	for (std::ptrdiff_t idx = 0; idx < nvox; ++idx)
		vox[idx] = final_id[vox[idx]];
	std::cout << "Done with remapping, total: " << (next_id - 1) << std::endl;
}


// A label's union-find parent and its count side by side: abiss's count in the low 31 bits and its
// ON_BORDER flag in bit 31 (packed only when every count fits, see counts_fit_packed).
struct PackedNode {
	Label parent;
	uint32_t count;
};
static const uint32_t PACKED_BORDER = 0x80000000u;

inline bool counts_fit_packed(const std::vector<std::size_t>& counts, std::size_t size_threshold) {
	if (counts.size() >= (std::size_t(1) << 31) || size_threshold >= (std::size_t(1) << 31))
		return false;
	for (const std::size_t c : counts)
		if ((c & ~ON_BORDER) >= (std::size_t(1) << 31))
			return false;
	return true;
}

// try_merge() on packed records: the same tests on the same numbers (a set flag puts a count at or
// above 2**31, as ON_BORDER puts it at or above 2**63, above any threshold), the same link and the
// same counts swap.
template <typename Nodes, typename Ranks>
inline void try_merge_packed(Nodes& uf, Ranks& rank, Label s1, Label s2,
		uint32_t size_threshold) {
	const uint32_t c1 = uf[s1].count, c2 = uf[s2].count;
	const uint32_t real1 = c1 & ~PACKED_BORDER, real2 = c2 & ~PACKED_BORDER;
	if (real1 >= size_threshold && real2 >= size_threshold)
		return;
	if (((PACKED_BORDER & (c1 | c2)) == 0) || (real1 >= size_threshold && c2 < size_threshold)
			|| (real2 >= size_threshold && c1 < size_threshold)) {
		uf[s1].count = (c1 + real2) | (c2 & PACKED_BORDER);
		uf[s2].count = 0;
		// boost's link_sets: union by rank, the second argument's root on a tie.
		Label root;
		if (rank[s1] > rank[s2]) {
			uf[s2].parent = s1;
			root = s1;
		} else {
			uf[s1].parent = s2;
			root = s2;
			if (rank[s1] == rank[s2])
				++rank[s2];
		}
		if (root != s1)
			std::swap(uf[root].count, uf[s1].count);
	} else {
		uf[s1].count = c1 | (c2 & PACKED_BORDER);
		uf[s2].count = c2 | (uf[s1].count & PACKED_BORDER);
	}
}

// The counts as packed records' counts: abiss's size in the low 31 bits, its ON_BORDER flag in bit 31.
inline std::vector<uint32_t> pack_counts(std::vector<std::size_t>& counts) {
	std::vector<uint32_t> packed(counts.size());
	for (std::size_t i = 0; i < counts.size(); ++i)
		packed[i] = (uint32_t)(counts[i] & ~ON_BORDER) | ((counts[i] & ON_BORDER) ? PACKED_BORDER : 0u);
	std::vector<std::size_t>().swap(counts);
	return packed;
}

// merge_segments() on packed records; consumes the edges and the counts.
inline void merge_segments_packed(LabelVolume& seg_raw, MappedBuffer<EdgePair>& edges, std::vector<uint32_t>& counts,
		std::size_t size_threshold_full) {
	const std::size_t n = counts.size();
	const uint32_t size_threshold = (uint32_t)size_threshold_full;
	MappedBuffer<PackedNode> uf;
	uf.resize_huge(n);
	MappedBuffer<uint8_t> rank;
	rank.resize_huge(n);
	{
		PackedNode* const u = uf.data();
		uint8_t* const rk = rank.data();
		const uint32_t* const c = counts.data();
#pragma omp parallel for num_threads(wz_threads()) schedule(static)
		for (std::ptrdiff_t i = 0; i < (std::ptrdiff_t)n; ++i) {
			u[i].parent = (Label)i;
			u[i].count = c[i];
			rk[i] = 0;
		}
	}
	std::vector<uint32_t>().swap(counts);
	const std::size_t ne = edges.size();
	const EdgePair* const ep = edges.data();
	const std::size_t BLOCK = std::size_t(1) << 16;
	// Records read this many edges ahead are asked into cache first: the edges are in affinity order, so
	// their labels' records are scattered over the whole table.
	const std::size_t PREFETCH = 16;
	const int nt = wz_threads();
	std::vector<Label> ra(BLOCK), rb(BLOCK);
	std::vector<uint8_t> live(BLOCK);
	PackedNode* const ufp = uf.data();
	// The root of x as the sets stand, its path pointed straight at it. While no thread links, every
	// node's root is fixed, so concurrent compressions write the same value.
	auto root_of = [ufp](Label x) -> Label {
		Label r = x;
		for (Label p; (p = __atomic_load_n(&ufp[r].parent, __ATOMIC_RELAXED)) != r;)
			r = p;
		while (x != r) {
			const Label p = __atomic_load_n(&ufp[x].parent, __ATOMIC_RELAXED);
			if (p != r)
				__atomic_store_n(&ufp[x].parent, r, __ATOMIC_RELAXED);
			x = p;
		}
		return r;
	};
	for (std::size_t e0 = 0; e0 < ne; e0 += BLOCK) {
		const std::size_t m = std::min(BLOCK, ne - e0);
		// The roots at the block's start, and whether the edge can still change anything: one set stays
		// one set, a root 0 stays 0, and two sets at or above the threshold are refused untouched.
#pragma omp parallel for num_threads(nt) schedule(static)
		for (std::ptrdiff_t i = 0; i < (std::ptrdiff_t)m; ++i) {
			if (e0 + i + PREFETCH < ne) {
				__builtin_prefetch(&ufp[ep[e0 + i + PREFETCH].lo]);
				__builtin_prefetch(&ufp[ep[e0 + i + PREFETCH].hi]);
			}
			const Label a = root_of(ep[e0 + i].lo);
			const Label b = root_of(ep[e0 + i].hi);
			ra[i] = a;
			rb[i] = b;
			live[i] = a != b && a && b
				&& !((ufp[a].count & ~PACKED_BORDER) >= size_threshold
					&& (ufp[b].count & ~PACKED_BORDER) >= size_threshold);
		}
		// In order: a provisional root linked by an earlier edge of this block is followed to the
		// root it now has, which is the root the serial loop's find gives.
		for (std::size_t i = 0; i < m; ++i) {
			if (i + PREFETCH < m && live[i + PREFETCH]) {
				__builtin_prefetch(&ufp[ra[i + PREFETCH]]);
				__builtin_prefetch(&ufp[rb[i + PREFETCH]]);
			}
			if (!live[i])
				continue;
			Label s1 = ra[i];
			while (uf[s1].parent != s1)
				s1 = uf[s1].parent;
			Label s2 = rb[i];
			while (uf[s2].parent != s2)
				s2 = uf[s2].parent;
			if (s1 != s2)
				try_merge_packed(uf, rank, s1, s2, size_threshold);
		}
	}
	edges.release();
	auto find = [&](Label x) -> Label {
		while (uf[x].parent != x) {
			const Label g = uf[uf[x].parent].parent;
			uf[x].parent = g;
			x = g;
		}
		return x;
	};
	MappedBuffer<Label> remaps;
	remaps.resize_huge(n);
	std::fill(remaps.data(), remaps.data() + n, Label(0));
	uf[0].count &= ~PACKED_BORDER;
	remaps[0] = 0;
	Label next_id = 1;
	for (std::size_t id = 0; id < n; ++id) {
		const Label s = find((Label)id);
		if (uf[id].count & ~PACKED_BORDER) {
			if (s && (uf[s].count >= size_threshold)) {
				if (remaps[s] == 0) {
					remaps[s] = next_id;
					uf[next_id].count = uf[s].count & ~PACKED_BORDER;
					++next_id;
				}
			} else {
				uf[s].count = 0;
				remaps[s] = 0;
			}
		}
	}
	// Every label's new id, through its root, once per label, into the record's count (no longer read).
	const Label* const rm = remaps.data();
#pragma omp parallel for num_threads(nt) schedule(static)
	for (std::ptrdiff_t id = 0; id < (std::ptrdiff_t)n; ++id) {
		Label x = (Label)id;
		while (ufp[x].parent != x)
			x = ufp[x].parent;
		ufp[id].count = rm[x];
	}
	remaps.release();
	const PackedNode* const table = uf.data();
	Label* const vox = seg_raw.data();
	const std::ptrdiff_t nvox = (std::ptrdiff_t)seg_raw.size();
#pragma omp parallel for num_threads(wz_threads()) schedule(static)
	for (std::ptrdiff_t idx = 0; idx < nvox; ++idx)
		vox[idx] = table[vox[idx]].count;
	std::cout << "Done with remapping, total: " << (next_id - 1) << std::endl;
}

// The whole of abiss.watershed(): labels for the unpadded volume, C-contiguous (Z, Y, X).
template <typename Code>
inline void run(const Code* codes, const float* lut, std::size_t Z, std::size_t Y, std::size_t X,
		float low, float high, std::size_t size_threshold, uint32_t* out) {
	PaddedAffs<Code> aff{codes, lut, (std::ptrdiff_t)X, (std::ptrdiff_t)Y, (std::ptrdiff_t)Z};
	LabelVolume seg_raw;
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
		const bool list_graph = nonnegative && (std::size_t)3 * X * Y * Z < (std::size_t)UINT32_MAX;
		if (list_graph && counts_fit_packed(counts, size_threshold)) {
			std::vector<uint32_t> packed = pack_counts(counts);
			// The scan in z slabs across threads when the volume is thick enough
			// (region_graph_lean_slabs: labels are numbered in scan order, so each slab's own labels
			// are one range and its edges with an own larger label cannot have occurred before it;
			// edges between two labels that both reach a later slab are matched across slabs).
			MappedBuffer<EdgePair> edges;
			if (!region_graph_lean_slabs(aff, seg_raw, packed.size() - 1, low, edges))
				edges = region_graph_lean(aff, seg_raw, packed.size() - 1, low);
			merge_segments_packed(seg_raw, edges, packed, size_threshold);
		} else {
		if (list_graph) {
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
		merge_segments_reference(seg_raw, rg, counts, size_threshold);
		}
	}
	const std::ptrdiff_t xdim = X + 2, ydim = Y + 2;
#pragma omp parallel for num_threads(wz_threads()) schedule(static)
	for (std::ptrdiff_t z = 0; z < (std::ptrdiff_t)Z; ++z)
		for (std::size_t y = 0; y < Y; ++y) {
			const Label* row = &seg_raw[1 + xdim * ((y + 1) + ydim * (z + 1))];
			std::copy(row, row + X, out + (z * Y + y) * X);
		}
}

}  // namespace abiss_ws

#endif  // WATERZ_ABISS_WATERSHED_H__
