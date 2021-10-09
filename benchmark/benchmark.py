from udnn import Model, Conv2D, Flatten, Dense
import time
import tensorflow as tf
import numpy as np


def build_udnn_model(input_size, num_class, num_filters, kernel_size, dtype):
    model = Model()
    model.add_layer("conv0", Conv2D(input_size, dtype, kernel_size,
                                    num_filters))
    model.add_layer("conv1", Conv2D(model.get_layer(0).out.shape[:3], dtype,
                                    kernel_size, num_filters))
    model.add_layer("flatten0", Flatten(model.get_layer(1).out.shape[:3], dtype))
    model.add_layer("dense0", Dense(model.get_layer(2).out.shape[:3], dtype, num_class))
    return model


def build_tf_model(input_size, num_class, num_filters, kernel_size, dtype):
    # setup reference model
    ref_model = tf.keras.Sequential()
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
    return ref_model


def copy_weights(ref_model, udnn_model):
    ref_weights = ref_model.weights
    udnn_model.load_weights(ref_weights)


def main():
    # CIFAR-10 uses 32x32x3 image
    input_size = (32, 32, 3)
    # CIFAR-10 has 10 classes
    num_class = 10
    # set input
    num_filters = 32
    kernel_size = 3
    dtype = "float32"
    num_runs = 10

    udnn_model = build_udnn_model(input_size, num_class, num_filters, kernel_size, dtype)
    ref_model = build_tf_model(input_size, num_class, num_filters, kernel_size, dtype)
    copy_weights(ref_model, udnn_model)

    input_vector = np.ones(input_size, dtype=dtype)
    input_ref_vector = input_vector.reshape([1] + list(input_size))

    start = time.time()
    for i in range(num_runs):
        ref_model.predict(input_ref_vector)
    end = time.time()
    ref_time = (end - start) / num_runs * 1000

    start = time.time()
    for i in range(num_runs):
        udnn_model.predict(input_vector)
    end = time.time()
    udnn_time = (end - start) / num_runs * 1000

    print("Tensorflow:", ref_time)
    print("uDNN:", udnn_time)


if __name__ == "__main__":
    main()
