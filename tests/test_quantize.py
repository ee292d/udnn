from udnn import quantize, unquantize, quantize_add, quantize_mult, ReLu, Dense, Conv2D, MaxPooling
from _udnn import float2fix_int16

import random
import tensorflow as tf
import numpy as np


def test_quantize():
    a = 10
    # we assume [s, e, e, e, m, m, m, m], i.e. Q3.4
    quantize_bias = 0x20  # 2 in fixed point
    quantize_scale = 0x40  # 4 in fixed point
    q = quantize(a, quantize_bias, quantize_scale, dtype="int8")
    # and truncation rounding
    assert q == 0x20  # 2 in fixed point


def test_unquantize():
    q = 0x20
    # we assume [s, e, e, e, m, m, m, m], i.e. Q3.4
    quantize_bias = 0x20  # 2 in fixed point
    quantize_scale = 0x40  # 4 in fixed point
    a = unquantize(q, quantize_bias, quantize_scale, dtype="int8")
    assert a == 10


def test_quantization():
    for i in range(50, 50):
        # we assume [s, e, e, e, m, m, m, m], i.e. Q3.4
        quantize_bias = 0x20
        quantize_scale = 0x40
        q = quantize(i, quantize_bias, quantize_scale, dtype="int16")
        a = unquantize(q, quantize_bias, quantize_scale, dtype="int16")
        assert i == a


def test_quantization_add():
    quantize_bias = float2fix_int16(2)  # 2 in fixed point
    quantize_scale = float2fix_int16(4)  # 4 in fixed point
    for i in range(-10, 10):
        for j in range(-10, 10):
            for f in range(1, 4):
                num1 = i + 0.5 ** f
                num2 = j + 0.5 ** f * 3
                res1 = num1 + num2
                num1_q = quantize(num1, quantize_bias, quantize_scale, dtype="int16")
                num2_q = quantize(num2, quantize_bias, quantize_scale, dtype="int16")
                res_q = quantize_add(num1_q, num2_q, quantize_bias, quantize_scale, dtype="int16")
                res2 = unquantize(res_q, quantize_bias, quantize_scale, dtype="int16")
                # they should be exact
                assert res2 == res1


def test_quantization_mult():
    quantize_bias = float2fix_int16(2)  # 2 in fixed point
    quantize_scale = float2fix_int16(0.25)  # 0.25 in fixed point
    for i in range(-4, 4):
        for j in range(-4, 4):
            for f in range(1, 5):
                num1 = i + 0.5 ** f
                num2 = j
                res1 = num1 * num2
                num1_q = quantize(num1, quantize_bias, quantize_scale, dtype="int16")
                num2_q = quantize(num2, quantize_bias, quantize_scale, dtype="int16")
                res_q = quantize_mult(num1_q, num2_q, quantize_bias, quantize_scale, dtype="int16")
                res2 = unquantize(res_q, quantize_bias, quantize_scale, dtype="int16")
                # they should be exact
                assert res2 == res1


def test_max_pool2d_quantized():
    for i in range(10):
        random.seed(i)
        np.random.seed(i)
        model = tf.keras.Sequential()
        pool_size = 2
        model.add(tf.keras.layers.MaxPool2D((pool_size, pool_size)))
        batch_size = 1
        input_size = 8
        input_channel = 3
        total_size = input_size * input_size * input_channel
        input_tensor = np.reshape(
            np.array([random.randint(-127, 127) for _ in range(total_size)],
                     dtype="float32"), (batch_size, input_size, input_size, input_channel))
        tf_out = model.predict(input_tensor)

        maxpool = MaxPooling((input_size, input_size, input_channel), "int16", 2)
        quantized_tensor = np.reshape(input_tensor, (input_size, input_size, input_channel))
        bias = float2fix_int16(1)
        scale = float2fix_int16(2)
        quantized_tensor = quantize(quantized_tensor, bias, scale, dtype="int16")
        maxpool.quantization_bias = bias
        maxpool.quantization_scale = scale
        maxpool.forward_quantized(quantized_tensor)
        out = unquantize(maxpool.out, bias, scale)
        out = np.array(out)
        out = out.reshape((1, input_size // pool_size,
                           input_size // pool_size,
                           input_channel))

        # should be exact
        assert np.equal(out, tf_out).all()


def test_relu_quantized():
    for i in range(10):
        random.seed(i)
        np.random.seed(i)
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.ReLU())
        batch_size = 1
        input_size = 8
        input_channel = 3
        total_size = input_size * input_size * input_channel
        input_tensor = np.reshape(
            np.array([random.randint(-127, 127) for _ in range(total_size)],
                     dtype="float32"), (batch_size, input_size, input_size, input_channel))
        tf_out = model.predict(input_tensor)
        assert tf_out.shape == (batch_size, input_size, input_size, input_channel)

        relu = ReLu((input_size, input_size, input_channel), dtype="int16")
        assert relu.out_size == (input_size, input_size, input_channel, 1)
        quantized_tensor = np.reshape(input_tensor, (input_size, input_size, input_channel))
        bias = float2fix_int16(1)
        scale = float2fix_int16(2)
        quantized_tensor = quantize(quantized_tensor, bias, scale, dtype="int16")
        relu.quantization_bias = bias
        relu.quantization_scale = scale
        relu.forward_quantized(quantized_tensor)
        out = unquantize(relu.out, bias, scale)
        out = np.array(out)
        assert out.shape == (input_size, input_size, input_channel, 1)
        out = out.reshape((1, input_size, input_size, input_channel))

        # should be exact
        assert np.equal(out, tf_out).all()


def test_dense_quantized():
    for i in range(10):
        random.seed(i)
        np.random.seed(i)
        model = tf.keras.Sequential()
        # default linear, which is what we're implementing
        # no bias
        input_shape = 16
        output_shape = 8
        model.add(tf.keras.layers.Dense(output_shape, input_shape=(input_shape,),
                                        dtype="float32"))
        # set all weights to known numbers
        model.layers[0].set_weights(
            [np.array(np.random.randint(low=-5, high=5, size=model.weights[0].shape, dtype="int16"),
                      dtype="float32"),
             np.array(np.random.randint(low=-5, high=5, size=model.weights[1].shape, dtype="int16"),
                      dtype="float32")])

        input_tensor = np.ones((1, input_shape), dtype="float32")
        tf_out = model.predict(input_tensor)

        layer = Dense((1, input_shape, 1), "int16", output_shape)
        weights = model.weights[0].numpy().reshape(layer.weights.shape)
        bias_weights = model.weights[1].numpy().reshape(layer.bias.shape)
        bias = float2fix_int16(1)
        scale = float2fix_int16(2)

        quantized_tensor = np.reshape(input_tensor, (1, input_shape, 1))
        quantized_tensor = quantize(quantized_tensor, bias, scale, dtype="int16")

        weights = quantize(weights, bias, scale, dtype="int16")
        bias_weights = quantize(bias_weights, bias, scale, dtype="int16")
        layer.load_weights(weights)
        layer.load_bias(bias_weights)
        layer.quantization_bias = bias
        layer.quantization_scale = scale
        layer.forward_quantized(quantized_tensor)

        out = unquantize(layer.out, bias, scale)
        out = np.array(out)
        out = out.reshape(tf_out.shape)

        # should be exact
        assert np.equal(out, tf_out).all()


def test_con2d_quantized():
    for i in range(10):
        random.seed(i)
        np.random.seed(i)
        input_size = 4
        num_filter = 10
        kernel_size = 3
        input_channel = 1
        input_shape = (1, input_size, input_size, input_channel)
        model = tf.keras.Sequential()
        conv = tf.keras.layers.Conv2D(num_filter, (kernel_size, kernel_size),
                                      dtype="float32")
        model.add(conv)
        model.build(input_shape)
        # set all weights to known numbers
        model.layers[0].set_weights(
            [np.array(np.random.randint(low=-5, high=5, size=model.weights[0].shape, dtype="int16"),
                      dtype="float32"),
             np.array(np.random.randint(low=-5, high=5, size=model.weights[1].shape, dtype="int16"),
                      dtype="float32")])

        input_tensor = np.ones((1, input_size, input_size, input_channel), dtype="float32")
        tf_out = model.predict(input_tensor)
        tf_out = tf_out.reshape((input_size - kernel_size + 1,
                                 input_size - kernel_size + 1, num_filter))

        layer = Conv2D((input_size, input_size, input_channel), "int16", kernel_size, num_filter)
        weights = model.weights[0].numpy().reshape(layer.weights.shape)
        bias_weights = model.weights[1].numpy().reshape(layer.bias.shape)
        bias = float2fix_int16(1)
        scale = float2fix_int16(2)
        weights = quantize(weights, bias, scale, dtype="int16")
        bias_weights = quantize(bias_weights, bias, scale, dtype="int16")
        layer.load_weights(weights)
        layer.load_bias(bias_weights)
        layer.quantization_bias = bias
        layer.quantization_scale = scale

        quantized_tensor = np.reshape(input_tensor, (input_size, input_size, input_channel))
        quantized_tensor = quantize(quantized_tensor, bias, scale, dtype="int16")
        layer.forward_quantized(quantized_tensor)
        out = unquantize(layer.out, bias, scale)
        out = np.array(out)
        out = out.reshape(tf_out.shape)

        # should be exact
        assert np.isclose(out, tf_out).all()


if __name__ == "__main__":
    test_quantization_add()
