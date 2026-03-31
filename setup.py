import sys
import tempfile
from pathlib import Path

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

include_dirs = [
    "src/waterz",
    "src/waterz/backend",
    numpy.get_include(),
]
if sys.platform == "darwin":
    include_dirs.append("/opt/homebrew/include")  # homebrew ... for boost

evaluate = Extension(
    name="waterz.evaluate",
    sources=[
        "src/waterz/evaluate.pyx",
        "src/waterz/frontend_evaluate.cpp",
    ],
    include_dirs=include_dirs,
    language="c++",
    extra_link_args=["-std=c++11"],
    extra_compile_args=["-std=c++11", "-w"],
)

# Pre-compile default agglomeration module (OneMinus<MeanAffinity>, PriorityQueue)
# so it doesn't need JIT compilation at runtime.
_generated_dir = Path(tempfile.mkdtemp())
(_generated_dir / "ScoringFunction.h").write_text(
    "typedef OneMinus<MeanAffinity<RegionGraphType, ScoreValue>> ScoringFunctionType;"
)
(_generated_dir / "Queue.h").write_text(
    "template<typename T, typename S> using QueueType = PriorityQueue<T, S>;"
)

agglomerate = Extension(
    name="waterz._agglomerate_default",
    sources=[
        "src/waterz/agglomerate.pyx",
        "src/waterz/frontend_agglomerate.cpp",
    ],
    include_dirs=include_dirs + [str(_generated_dir)],
    language="c++",
    extra_link_args=["-std=c++11"],
    extra_compile_args=["-std=c++11", "-w", "-O3"],
)

setup(ext_modules=cythonize([evaluate, agglomerate]))
