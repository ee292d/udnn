import subprocess

from models import build_tf_model, build_udnn_model
from tensorflow import keras

import os
import shutil


def copy_weights(ref_model, udnn_model):
    ref_weights = ref_model.weights
    udnn_model.load_weights(ref_weights)


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(root)
    submission_dir = os.path.join(project_root, "submission")
    checkpoint = os.path.join(root, "ref_model")
    # CIFAR-10 uses 32x32x3 image
    input_size = (32, 32, 3)
    # CIFAR-10 has 10 classes
    num_class = 10

    tf_model = keras.models.load_model(checkpoint)
    udnn_model = build_udnn_model(input_size, num_class, 3, "float32")
    copy_weights(tf_model, udnn_model)

    # feel free to change this number
    quantization_bias = 128  # 0
    quantization_scale = 256  # 1
    quantized_model = udnn_model.quantize(quantization_bias, quantization_scale, dtype="int16")
    quantized_model.dump_weights_to_dir(submission_dir)
    # copy files
    src_root = os.path.join(project_root, "src")
    for filename in ["layer.hh", "quantize.hh"]:
        filename = os.path.join(src_root, filename)
        shutil.copy2(filename, submission_dir)

    # zip the entire thing up
    dst_zip = os.path.join(project_root, "submission.zip")
    if os.path.exists(dst_zip):
        shutil.rmtree(dst_zip, ignore_errors=True)
    subprocess.check_call(["zip", "-r", dst_zip, submission_dir])


if __name__ == "__main__":
    main()
