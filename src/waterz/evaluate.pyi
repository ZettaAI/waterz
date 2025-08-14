from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

class Metrics(TypedDict):
    voi_split: float
    voi_merge: float
    rand_split: float
    rand_merge: float

def evaluate(segmentation: NDArray[np.uint64], gt: NDArray[np.uint64]) -> Metrics: ...
