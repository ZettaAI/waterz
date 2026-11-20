from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("waterz")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "uninstalled"

from ._agglomerate import (
    agglomerate,
    agglomerate_shell,
    build_region_graph_only,
    watershed_codes,
    build_region_graph_rich,
    get_num_threads,
    set_num_threads,
)
from ._merge import (
    dust_merge_from_region_graph,
    get_region_graph,
    get_region_graph_rich,
    merge_dust,
    merge_function_to_scoring,
    merge_region_graphs,
    merge_segments,
    strip_boundary,
)
from .evaluate import evaluate

__all__ = [
    "agglomerate",
    "agglomerate_shell",
    "watershed_codes",
    "build_region_graph_only",
    "build_region_graph_rich",
    "dust_merge_from_region_graph",
    "evaluate",
    "get_num_threads",
    "get_region_graph",
    "get_region_graph_rich",
    "merge_dust",
    "merge_function_to_scoring",
    "merge_region_graphs",
    "merge_segments",
    "set_num_threads",
    "strip_boundary",
]
