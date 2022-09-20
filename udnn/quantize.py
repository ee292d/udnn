from _udnn import quantize as _quantize, unquantize as _unquantize, TensorBase, quantize_int8, quantize_int16, \
    TensorFloat, unquantize_int8, unquantize_int16

import numpy as np


def quantize(value, bias, scale, dtype="int8"):
    assert dtype in {"int8", "int16"}
    if isinstance(value, TensorBase):
        if dtype == "int8":
            return quantize_int8(value, bias, scale)
        else:
            return quantize_int16(value, bias, scale)
    elif hasattr(value, "shape"):
        ref_dtype = value.dtype
        assert ref_dtype == "float32", "Only floats supported"
        value = TensorFloat(value)
        if dtype == "int8":
            return quantize_int8(value, bias, scale)
        else:
            return quantize_int16(value, bias, scale)
    else:
        return _quantize(value, bias, scale, dtype)


def unquantize(value, bias, scale, dtype="int8"):
    assert dtype in {"int8", "int16"}
    if isinstance(value, TensorBase):
        dtype = value.dtype
        if dtype == "int8":
            return unquantize_int8(value, bias, scale)
        else:
            return unquantize_int16(value, bias, scale)
    elif hasattr(value, "shape"):
        ref_dtype = value.dtype
        assert ref_dtype == "float32", "Only floats supported"
        value = TensorFloat(value)
        if dtype == "int8":
            return unquantize_int8(value, bias, scale)
        else:
            return unquantize_int16(value, bias, scale)
    else:
        return _unquantize(value, bias, scale, dtype)
