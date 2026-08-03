from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("waterz")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "uninstalled"

from ._agglomerate import agglomerate, build_region_graph_only, build_region_graph_rich
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
    "build_region_graph_only",
    "build_region_graph_rich",
    "dust_merge_from_region_graph",
    "evaluate",
    "get_region_graph",
    "get_region_graph_rich",
    "merge_dust",
    "merge_function_to_scoring",
    "merge_region_graphs",
    "merge_segments",
    "strip_boundary",
]
