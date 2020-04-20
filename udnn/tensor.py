from _udnn import TensorInt8, TensorInt16, TensorInt32, TensorInt64, \
    TensorFloat, TensorDouble


def tensor(shape, dtype):
    dtype = dtype.lower()
    if dtype == "int8":
        return TensorInt8(shape)
    elif dtype == "int16":
        return TensorInt16(shape)
    elif dtype == "int32":
        return TensorInt32(shape)
    elif dtype == "int64":
        return TensorInt64(shape)
    elif dtype == "float32" or dtype == "float":
        return TensorFloat(shape)
    elif dtype == "double" or dtype == "float64":
        return TensorDouble(shape)
    else:
        raise TypeError("Unrecognized dtype " + dtype)
