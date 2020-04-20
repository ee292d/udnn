import numpy as np
from udnn import tensor
import pytest


@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "float32",
                                   "float64"])
def test_buffer_convert(dtype):
    data = [i for i in range(4 * 4 * 3)]

    a = np.array(data, dtype=dtype)
    a = a.reshape((4, 4, 3, 1))
    b = tensor(a, dtype=dtype)
    d = np.array(b, dtype=dtype)
    for x in range(4):
        for y in range(4):
            for c in range(3):
                assert b[x, y, c] == a[x, y, c]
                assert d[x, y, c] == a[x, y, c]


def test_shape():
    shape = (1, 2, 3)
    a = np.ones(shape=shape, dtype="int8")
    b = tensor(shape, dtype="int8")
    assert b.shape != a.shape
    assert b.shape == (1, 2, 3, 1)


if __name__ == "__main__":
    test_shape()
