from udnn import Model, Conv2D, Flatten, Dense, ReLu, MaxPooling
import tensorflow as tf


def build_udnn_model(input_size, num_class, kernel_size, dtype):
    model = Model()
    conv0 = model.add_layer("conv0", Conv2D(input_size, dtype, kernel_size,
                                            32))
    relu0 = model.add_layer("relu0", ReLu(conv0.out.shape[:3], dtype))
    max_pool0 = model.add_layer("max_pool0", MaxPooling(relu0.out.shape[:3], dtype, 2))

    conv1 = model.add_layer("conv1", Conv2D(max_pool0.out.shape[:3], dtype, kernel_size,
                                            64))
    relu1 = model.add_layer("relu1", ReLu(conv1.out.shape[:3], dtype))
    max_pool1 = model.add_layer("max_pool1", MaxPooling(relu1.out.shape[:3], dtype, 2))

    conv2 = model.add_layer("conv2", Conv2D(max_pool1.out.shape[:3], dtype, kernel_size,
                                            64))
    relu2 = model.add_layer("relu2", ReLu(conv2.out.shape[:3], dtype))
    flatten0 = model.add_layer("flatten0", Flatten(relu2.out.shape[:3], dtype))
    dense0 = model.add_layer("dense0", Dense(flatten0.out.shape[:3], dtype, 64))
    relu3 = model.add_layer("relu3", ReLu(dense0.out.shape[:3], dtype))
    model.add_layer("dense1", Dense(relu3.out.shape[:3], dtype, num_class))
    return model


def build_tf_model(input_size, num_class, kernel_size, dtype):
    # setup reference model
    ref_model = tf.keras.Sequential()
    ref_model.add(tf.keras.layers.Input(input_size, 1, dtype=dtype))
    ref_model.add(tf.keras.layers.Conv2D(32, kernel_size,
                                         bias_initializer="random_uniform",
                                         dtype=dtype))
    ref_model.add(tf.keras.layers.ReLU())
    ref_model.add(tf.keras.layers.MaxPool2D((2, 2)))
    ref_model.add(tf.keras.layers.Conv2D(64, kernel_size,
                                         bias_initializer="random_uniform",
                                         dtype=dtype))
    ref_model.add(tf.keras.layers.ReLU())
    ref_model.add(tf.keras.layers.MaxPool2D((2, 2)))
    ref_model.add(tf.keras.layers.Conv2D(64, kernel_size,
                                         bias_initializer="random_uniform",
                                         dtype=dtype))
    ref_model.add(tf.keras.layers.ReLU())
    ref_model.add(tf.keras.layers.Flatten())
    ref_model.add(tf.keras.layers.Dense(64, dtype=dtype))
    ref_model.add(tf.keras.layers.Dense(num_class, dtype=dtype))
    ref_model.build()
    return ref_model