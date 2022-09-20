from udnn import quantize, unquantize
from tensorflow.keras import datasets
from models import build_udnn_model
import time
import numpy as np
import tensorflow as tf
import argparse


def get_test_image():
    (_, __), (test_images, test_labels) = datasets.cifar10.load_data()
    # need to normalize the images
    test_images = test_images / 255.0
    return test_images, test_labels


def get_args():
    parser = argparse.ArgumentParser("Benchmark udnn")
    parser.add_argument("weights", type=str)
    return parser.parse_args()


def main():
    # CIFAR-10 uses 32x32x3 image
    input_size = (32, 32, 3)
    # CIFAR-10 has 10 classes
    num_class = 10
    # set input
    kernel_size = 3
    dtype = "float32"
    quantized_dtype = "int16"

    args = get_args()
    weight_dir = args.weights

    udnn_model = build_udnn_model(input_size, num_class, kernel_size, quantized_dtype)
    udnn_model.load_weights_from_dir(weight_dir)

    # read quantization from the model
    bias = udnn_model.get_layer(0).quantization_bias
    scale = udnn_model.get_layer(0).quantization_scale

    test_images, test_labels = get_test_image()

    num_test = 100
    test_images = test_images[:num_test]
    test_labels = test_labels[:num_test]

    total_time = 0
    for i in range(len(test_labels)):
        image = np.array(test_images[i], dtype=dtype)
        image = quantize(image, bias, scale, dtype=quantized_dtype)
        start = time.time()
        udnn_model.predict(image)
        t = time.time() - start
        total_time += t
    print("Average time:", total_time / num_test)


if __name__ == "__main__":
    main()
