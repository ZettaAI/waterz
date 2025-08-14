import builtins
import os

from Cython.Build import cythonize
from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.extension import Extension


class build_ext(_build_ext):
    """Ensure numpy headers are available during build."""

    def finalize_options(self):
        _build_ext.finalize_options(self)
        builtins.__NUMPY_SETUP__ = False
        import numpy

        self.include_dirs.append(numpy.get_include())


here = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(here, "src", "waterz")
include_dirs = [
    source_dir,
    os.path.join(source_dir, "backend"),
    # Common Homebrew include paths (Intel + Apple Silicon)
    "/usr/local/include",
    "/opt/homebrew/include",
]

extensions = cythonize(
    [
        Extension(
            name="waterz.evaluate",
            sources=[
                os.path.join("src", "waterz", "evaluate.pyx"),
                os.path.join("src", "waterz", "frontend_evaluate.cpp"),
            ],
            include_dirs=include_dirs,
            language="c++",
            extra_link_args=["-std=c++11"],
            extra_compile_args=["-std=c++11", "-w"],
        )
    ],
    language_level=3,
)


setup(
    cmdclass={"build_ext": build_ext},
    ext_modules=extensions,
)
