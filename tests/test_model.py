from udnn import Model, Conv2D, Dense, Flatten
import tempfile
import numpy as np
import tensorflow as tf


def build_model():
    model = Model()
    # All conv
    model.add_layer("conv0", Conv2D((64, 64, 3), "int8", 3, 2))
    model.add_layer("conv1", Conv2D((62, 62, 2), "int8", 3, 1))
    return model


def test_model_load_dump():
    with tempfile.TemporaryDirectory() as temp:
        model0 = build_model()
        # generate random weights
        conv0_weight_size = model0.get_layer(0).weights_size.tuple()
        conv0_bias_size = model0.get_layer(0).bias_size.tuple()
        conv0_weights = np.random.randint(-128, high=128, size=conv0_weight_size, dtype="int8")
        conv0_bias = np.random.randint(-128, high=128, size=conv0_bias_size, dtype="int8")
        conv1_weight_size = model0.get_layer(1).weights_size.tuple()
        conv1_bias_size = model0.get_layer(1).bias_size.tuple()
        conv1_weights = np.random.randint(-128, high=128, size=conv1_weight_size, dtype="int8")
        conv1_bias = np.random.randint(-128, high=128, size=conv1_bias_size, dtype="int8")
        model0.load_weights([conv0_weights, conv0_bias, conv1_weights, conv1_bias])
        # quantization
        model0.get_layer(0).quantization_bias = 42
        model0.get_layer(0).quantization_scale = -42
        model0.is_quantized = True
        model0.dump_weights_to_dir(temp)

        # load the weights up
        model1 = build_model()
        model1.load_weights_from_dir(temp)
        # make sure they have the same weights

        conv0_w = model1.get_layer(0).weights
        conv1_w = model1.get_layer(1).weights
        assert np.equal(np.array(conv0_w), conv0_weights).all()
        assert np.equal(np.array(conv1_w), conv1_weights).all()
        assert model0.get_layer(0).quantization_bias == model1.get_layer(0).quantization_bias
        assert model0.get_layer(0).quantization_scale == model1.get_layer(0).quantization_scale
        assert model1.is_quantized


def test_model_predict():
    # CIFAR-10 uses 32x32x3 image
    input_size = (32, 32, 3)
    # CIFAR-10 has 10 classes
    num_class = 10
    # set input
    num_filters = 32
    kernel_size = 3

    # setup reference model
    ref_model = tf.keras.Sequential()
    # notice that normal tensorflow only takes float for conv2d layer
    dtype = "float32"
    ref_model.add(tf.keras.layers.Input(input_size, 1, dtype=dtype))
    ref_model.add(tf.keras.layers.Conv2D(num_filters, kernel_size,
                                         bias_initializer="random_uniform",
                                         dtype=dtype))
    ref_model.add(tf.keras.layers.Conv2D(num_filters, kernel_size,
                                         bias_initializer="random_uniform",
                                         dtype=dtype))
    ref_model.add(tf.keras.layers.Flatten())
    ref_model.add(tf.keras.layers.Dense(num_class))
    ref_model.build()

    # this is our implementation
    model = Model()
    model.add_layer("conv0", Conv2D(input_size, dtype, kernel_size,
                                    num_filters))
    model.add_layer("conv1", Conv2D(model.get_layer(0).out.shape[:3], dtype,
                                    kernel_size, num_filters))
    model.add_layer("flatten0", Flatten(model.get_layer(1).out.shape[:3], dtype))
    model.add_layer("dense0", Dense(model.get_layer(2).out.shape[:3], dtype, num_class))

    ref_weights = ref_model.weights
    model.load_weights(ref_weights)

    # test inputs
    input_vector = np.ones(input_size, dtype=dtype)
    ref_out = ref_model.predict(input_vector.reshape([1] + list(input_size)))
    model_out = np.array(model.predict(input_vector))
    model_out = model_out.reshape(ref_out.shape)
    assert np.isclose(model_out, ref_out, atol=1.e-5).all()


if __name__ == "__main__":
    test_model_predict()
