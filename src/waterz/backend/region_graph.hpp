#pragma once

#include "types.hpp"

#include <cstddef>
#include <iostream>

/**
 * Extract the region graph from a segmentation. Edges are annotated with the 
 * maximum affinity between the regions.
 *
 * @param aff [in]
 *              The affinity graph to read the affinities from.
 * @param seg [in]
 *              The segmentation.
 * @param max_segid [in]
 *              The highest ID in the segmentation.
 * @param statisticsProvider [in]
 *              A statistics provider to update on-the-fly.
 * @param region_graph [out]
 *              A reference to a region graph to store the result.
 */
template<typename AG, typename V, typename StatisticsProviderType>
inline
void
get_region_graph(
		const AG& aff,
		const V& seg,
		std::size_t max_segid,
		StatisticsProviderType& statisticsProvider,
		RegionGraph<typename V::element>& rg) {

	typedef typename AG::element F;
	typedef typename V::element ID;
	typedef RegionGraph<ID> RegionGraphType;
	typedef typename RegionGraphType::EdgeIdType EdgeIdType;

	std::ptrdiff_t zdim = aff.shape()[1];
	std::ptrdiff_t ydim = aff.shape()[2];
	std::ptrdiff_t xdim = aff.shape()[3];

	std::size_t p[3];
	for (p[0] = 0; p[0] < zdim; ++p[0])
		for (p[1] = 0; p[1] < ydim; ++p[1])
			for (p[2] = 0; p[2] < xdim; ++p[2]) {

				ID id1 = seg[p[0]][p[1]][p[2]];
				statisticsProvider.addVoxel(id1, p[2], p[1], p[0]);

				for (int d = 0; d < 3; d++) {

					if (p[d] == 0)
						continue;

					ID id2 = seg[p[0]-(d==0)][p[1]-(d==1)][p[2]-(d==2)];

					if (id1 != id2) {

						EdgeIdType e = rg.findEdge(id1, id2);
						if (e == RegionGraphType::NoEdge) {
							e = rg.addEdge(id1, id2);
							statisticsProvider.notifyNewEdge(e);
						}
						statisticsProvider.addAffinity(e, aff[d][p[0]][p[1]][p[2]]);
					}
				}
			}

	std::cout << "Region graph number of edges: " << rg.edges().size() << std::endl;
}

template<typename ID, typename StatisticsProviderType>
inline
void
initialize_with_region_graph(
		StatisticsProviderType& statisticsProvider,
		RegionGraph<ID>& rg,
		const std::vector<ScoredEdge>& edges,
		const std::vector<double>& edges_metadata) {

	typedef RegionGraph<ID> RegionGraphType;
	typedef typename RegionGraphType::EdgeIdType EdgeIdType;

	for (int i = 0; i < edges.size(); ++i) {
		const auto& edge = edges[i];
		double size = edges_metadata[i];
		auto aff = 1.0 - edge.score;
		EdgeIdType e = rg.addEdge(edge.u, edge.v);
		statisticsProvider.addEdge(e, aff, size);
	}
}
