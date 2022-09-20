import _udnn


def __import_layer(base_name, type_str):
    if type_str == "float32":
        type_str = "float"
    elif type_str == "float64":
        type_str = "double"
    type_str = type_str.capitalize()
    c = getattr(_udnn, base_name + type_str)
    return c


def __convert_shape_to_size(shape):
    assert isinstance(shape, (tuple, list))
    assert len(shape) == 3
    return _udnn.TensorSize(shape)


def __get_layer(shape, dtype, name, *args):
    shape = __convert_shape_to_size(shape)
    c = __import_layer(name + "Layer", dtype)
    return c(shape, *args)


def Flatten(shape, dtype, *args):
    return __get_layer(shape, dtype, "Flatten", *args)


def MaxPooling(shape, dtype, *args):
    return __get_layer(shape, dtype, "MaxPooling", *args)


def Conv2D(shape, dtype, *args):
    return __get_layer(shape, dtype, "Conv2D", *args)


def Dense(shape, dtype, *args):
    return __get_layer(shape, dtype, "Dense", *args)


def ReLu(shape, dtype, *args):
    return __get_layer(shape, dtype, "ReLuActivation", *args)
