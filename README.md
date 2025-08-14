[![License](https://img.shields.io/pypi/l/waterz.svg?color=green)](https://github.com/funkey/waterz/raw/master/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/waterz.svg?color=green)](https://pypi.org/project/waterz)
[![Python Version](https://img.shields.io/pypi/pyversions/waterz.svg?color=green)](https://python.org)
[![CI](https://github.com/funkey/waterz/actions/workflows/test.yml/badge.svg)](https://github.com/funkey/waterz/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/funkey/waterz/branch/main/graph/badge.svg?token=qGnz9GXpEb)](https://codecov.io/gh/funkey/waterz)

<!--
    the coerage is not working on c++ code, so the coverage is 0 now!
    [![Coverage Status](https://coveralls.io/repos/github/jingpengw/waterz/badge.svg?branch=master)](https://coveralls.io/github/jingpengw/waterz?branch=master)
-->

# waterz

Pronounced *water-zed*. A simple watershed and region agglomeration library for
affinity graphs.

Based on the watershed implementation of [Aleksandar Zlateski](https://bitbucket.org/poozh/watershed) and [Chandan Singh](https://github.com/TuragaLab/zwatershed).

# Installation

Install c++ dependencies:

```sh
sudo apt install libboost-dev
```

Install from PyPI

```sh
pip install waterz
```

install from local version

```sh
pip install .
```

# Usage

```python
import waterz
import numpy as np

# affinities is a [3,depth,height,width] numpy array of float32
affinities = ...

thresholds = [0, 100, 200]

segmentations = waterz.agglomerate(affinities, thresholds)
```

# Development

## Release to pypi

upgrade the version number in the `pyproject.toml` file, then

```sh
git tag v0.9.5
git push origin v0.9.5
```
