import tensorflow as tf
import numpy as np
import random
from udnn import Flatten, tensor, Dense, Conv2D, MaxPooling, ReLu


def test_flatten_simd():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    # batch size is always 1
    batch_size = 1
    total_size = 3 * 4 * 5
    input_tensor = np.reshape(
        np.array([random.randint(-127, 127) for i in range(total_size)],
                 dtype="int8"), (batch_size, 3, 4, 5))
    tf_out = model.predict(input_tensor)
    assert tf_out.shape == (1, 60)
    tf_out = tf_out.reshape(60)

    flatten = Flatten((3, 4, 5), dtype="int8")
    assert flatten.out_size == (1, 60, 1, 1)
    # notice that we don't deal with batch
    flatten.forward_simd(np.reshape(input_tensor, (3, 4, 5)))
    out = np.array(flatten.out)
    assert out.shape == (1, 60, 1, 1)
    out = out.reshape(60)
    # make sure they are equal
    # the following does not work
    # notice that the TF uses (batch, height, width, channels)
    assert np.equal(out, tf_out).all()


def test_dense_simd():
    tf.random.set_seed(0)
    model = tf.keras.Sequential()
    # default linear, which is what we're implementing
    # no bias
    input_shape = 16
    output_shape = 8
    model.add(tf.keras.layers.Dense(output_shape, input_shape=(input_shape,),
                                    dtype="float32",
                                    bias_initializer="random_uniform"))
    layer = Dense((1, input_shape, 1), "float32", output_shape)
    weights = model.weights[0].numpy().reshape(layer.weights.shape)
    bias = model.weights[1].numpy().reshape(layer.bias.shape)
    weights_t = tensor(weights, dtype="float32")
    bias_t = tensor(bias, dtype="float32")
    layer.load_weights(weights_t)
    layer.load_bias(bias_t)

    input_tensor = np.ones((1, input_shape), dtype="float32")
    tf_out = model.predict(input_tensor)
    # reshape it
    input_tensor = input_tensor.reshape((1, input_shape, 1))
    layer.forward_simd(input_tensor)
    out = np.reshape(layer.out, (1, output_shape))
    assert np.isclose(out, tf_out).all()


def test_conv_simd():
    tf.random.set_seed(0)
    input_size = 4
    num_filter = 10
    kernel_size = 3
    input_channel = 1
    input_shape = (1, input_size, input_size, input_channel)
    model = tf.keras.Sequential()
    conv = tf.keras.layers.Conv2D(num_filter, (kernel_size, kernel_size),
                                  dtype="float32",
                                  bias_initializer="random_uniform")
    model.add(conv)
    model.build(input_shape)

    weights = model.weights[0].numpy()
    layer = Conv2D((input_size, input_size, input_channel), "float32",
                   kernel_size, num_filter)
    weights_t = tensor(weights, dtype="float32")

    bias = model.weights[1].numpy().reshape(layer.bias.shape)
    bias_t = tensor(bias, dtype="float32")
    layer.load_weights(weights_t)
    layer.load_bias(bias_t)

    total_num = input_size * input_size * input_channel
    data = [i for i in range(total_num)]
    input_tensor = np.array(data, dtype="float32").reshape((1, input_size,
                                                            input_size,
                                                            input_channel))
    tf_out = model.predict(input_tensor)
    tf_out = tf_out.reshape((input_size - kernel_size + 1,
                             input_size - kernel_size + 1, num_filter))

    input_tensor = np.reshape(input_tensor,
                              (input_size, input_size, input_channel))
    layer.forward_simd(input_tensor)
    out = np.array(layer.out).reshape((input_size - kernel_size + 1,
                                       input_size - kernel_size + 1,
                                       num_filter))

    assert np.isclose(out, tf_out).all()


def test_max_pooling_simd():
    tf.random.set_seed(0)
    batch_size = 1
    pool_size = 2
    input_size = 32
    input_channel = 3
    input_shape = (1, input_size, input_size, input_channel)
    model = tf.keras.Sequential()
    pool = tf.keras.layers.MaxPool2D((pool_size, pool_size), dtype="float32")
    model.add(pool)
    model.build(input_shape)

    layer = MaxPooling((input_size, input_size, input_channel), "float32", pool_size)
    total_size = input_size * input_size * input_channel
    input_tensor = np.reshape(
        np.array([random.uniform(-100, 100) for _ in range(total_size)],
                 dtype="float32"), (batch_size, input_size, input_size, input_channel))
    tf_out = model.predict(input_tensor)
    input_tensor = np.reshape(input_tensor,
                              (input_size, input_size, input_channel))
    layer.forward_simd(input_tensor)
    out = np.array(layer.out).reshape((1, input_size // pool_size,
                                       input_size // pool_size,
                                       input_channel))
    assert np.isclose(out, tf_out).all()


def test_relu_simd():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.ReLU())
    batch_size = 1
    input_size = 8
    input_channel = 3
    total_size = input_size * input_size * input_channel
    input_tensor = np.reshape(
        np.array([random.randint(-127, 127) for _ in range(total_size)],
                 dtype="int8"), (batch_size, input_size, input_size, input_channel))
    tf_out = model.predict(input_tensor)
    assert tf_out.shape == (batch_size, input_size, input_size, input_channel)

    relu = ReLu((input_size, input_size, input_channel), dtype="int8")
    assert relu.out_size == (input_size, input_size, input_channel, 1)
    relu.forward_simd(np.reshape(input_tensor, (input_size, input_size, input_channel)))
    out = np.array(relu.out)
    assert out.shape == (input_size, input_size, input_channel, 1)
    out = out.reshape((1, input_size, input_size, input_channel))
    assert np.isclose(out, tf_out).all()


if __name__ == "__main__":
    test_relu()
