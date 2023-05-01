import collections.abc
import math
from itertools import repeat


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)


def calculate_output_dimension(w: int, k: int, p: int | str, s: int):
    # taken from: https://stackoverflow.com/questions/53580088/calculate-the-output-size-in-convolution-layer
    return math.floor((w - k + 2 * p) / s) + 1
