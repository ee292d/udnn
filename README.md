# EE292D Micro DNN (udnn)

## Overview
In this project, you will implement an inference-only DNN framework that is
capable of running various kinds of network architectures on CPU with native
instructions. You will be able to transfer your (potentially quantized) weights
from TensorFlow based models and use them in your own DNN framework.

## Getting started
We will be distributing assignments with git. Here is what you need to do to
get the starter code:
```
git clone --recurse-submodules -j2 https://github.com/ee292d/udnn.git
```
### Build instructions
This project has been thoroughly tested on Linux. You should be able to run it
on OSX, but we cannot guarantee support for that. If you do not have access
to a Linux machine, you can `ssh` into `cardinal.stanford.edu` to use
Stanford's linux machines. 

#### Prerequisite
- A C\++14 compiler: GCC 5.4+ or Clang++ 3.4+
- `cmake` 3.8+
- Python with development package installed

If you are on a debian based system, you can do
```
sudo apt install cmake g++ python3 python3-dev
```

Although the code works for Mac and Windows, we highly recommend you to work with a
Linux machine. You can use Stanford shared compute clusters such as `cardinal` if you
don't have direct access to a linux machine. Below are instructions on how to set it
up on `cardinal`.

1. SSH into a cardinal machine with your SUNet ID. Notice that there is a 2-step auth protection.
  ```
  ssh [your_sunet_id]@cardinal.stanford.edu
  ```
2. Setting up a virtual environnement called `env`:
  ```
  virtualenv --python=/usr/bin/python3 env
  ```
3. Activate the env
  ```
  source env/bin/activate
  ```
4. Download the tensorflow wheel without AVX instruction. This is necessary because the stanford PyPI
  wheel compiles against AVX but `cardinal` machines doesn't have one
  ```
  wget https://tf.novaal.de/barcelona/tensorflow-2.4.3-cp36-cp36m-linux_x86_64.whl
  ```
5. Install dependencies, which may take a while
  ```
  pip install cmake tensorflow-2.4.3-cp36-cp36m-linux_x86_64.whl
  ```
6. Test out if tensorflow has installed properly
  ```
  python -c "import tensorflow"
  ```

Part of the assignment requires you to set up udnn environment on EdgeTPU. Here are the instructions:
1. Make sure your coral board has internet access
2. Update the package list. Hit `y` if it prompts any error
   ```
   sudo apt update
   ```
3. Install `cmake`
   ```
   sudo apt install cmake
   ```

##### Build the native C++ code
Once you're in the root folder of the project, create a `build` folder and run cmake
```
mkdir build
cd build
cmake ..
```
Once `cmake` successfully generates the `Makefile`, do
```
make
```
This will build the entire project as well as the sample tests. To make sure it
is working, you can type
```
make test
```
Which runs some sample tests. Don't worry if some of them fail: it is part of
the project to make the tests green!

##### Build the Python binding
It is always a good idea to use some sort of virtual environment when using
Python to isolate different Python packages. To do so, you can use
`virtualenv` or any Python virtual environment package you prefer. If you haven't
done already, in the project root folder, do
```
virtualenv --python=python3 env
source env/bin/activate
```
You are now in the virtual environment and free to install any packages even
though you don't have root access. To install the Python binding, simply do
```
pip install -e .
```
Where `-e` allows to link the package to our root folder, which makes easy to
develop. To build the package in debug mode (see more details in the tips
section), you can do
```
DEBUG=1 pip install -e .
```
Your udnn code will run a little slower, but that allows you to attach gdb when
you debugging your code with Python-based tests. To install test dependencies,
simply do
```
pip install pytest keras tensorflow
```
If you have limited disk space, you can use `tensorflow==2.0` instead.
To make sure your Python binding is installed correctly, simply do
```
pytest tests/
```
You shall not see any alarming errors except that some tests fail.

## Starter code structure
The starter code includes two set of implementations: C++ for the core
functionality, and Python for helper class and tests against Tensorflow. As a
result, there are 4 main components in the project
- `src/*`: these are C++ files that you have to deal with most of the time.
- `python/*`: these are the Python binding code
- `tests/*`: these are the test code, in GoogleTest and `pytest`
- `udnn/*`: these are the Python helper code

### Tensor and Layers
In `src/tensor.hh`, we already provide an implementation of 4D tensor suitable
for most use cases. Notice that the dimension of the `Tensor` class is defined
by `TensorSize`, whose dimension is defined as `(y, x, c, k)`, where
`k` is default to `1` if not specified.

Since `Tensor` is a templated class, to instantiate a `Tensor` object, you need
to provide a type, such as `Tensor<int>`. To access the content, you can use
`auto a = t(y, x, c, k)` or `auto a = t(y, x, c)`, where t is the `Tensor`
object. To assign values to the tensor object, you can simply do
`t(y, x, c, k) = value`.

Defined in `src/layer.hh`, the `Layer` class is templated in the same way as
the `Layer` class. The base class has already populated some default
behaviors of the layer, such as whether it has weights or bias. The `forward`
function will be called to compute the actual activation. Notice that the
starter code already gives some common layer partial implementation to get you
familiar with the code.

### Model
The `Model` class is defined in `src/model.hh` and `src/model.cc`, it allows
different layers to be connected to each other as well as doing some sanity
check on the dimensionality. You can add more helper functions to it if the
change makes it easier for you to construct complex models, but modification
to this file is not required by this project.

### Python binding
`python/udnn.cc` defines a series of Python binding that's exposed as `_udnn`
in Python. We will cover some basic approaches to adding new layers to Python in
the section below. You can refer to this
[guide](https://pybind11.readthedocs.io/en/stable/index.html) if you want to
learn or experiment more about the Python bindings.

### Python helper code
`udnn/tensor.py` offers some utility functions to instantiate or convert numpy
array into tensor, the usage is
```Python
t = tensor((4, 3, 2, 1), "int8")
```
If we want to instantiate a `Tensor` with signed 8-bit byte and shape
`(4, 3, 2, 1)`.
Since we have implemented standard Python buffer protocol for you, you can also
pass in a numpy array with corresponding type:
```Python
array = np.ones((4, 3, 2, 1), dtype="int8")
t = tensor(array, dtype="int8")
```
Notice that the binding only takes either 3D or 4D array, you need to reshape
the numpy array when necessary.

`udnn/layer.py` offers some example code that illustrates how to instantiate a
C++ layer with `*args`, where `args` are the same constructor parameter in your
C++ code.

`udnn/model.py` allows you construct a model in Python as well as load weights
from or dump weights to a folder. It mimics some interface of the `keras` 
`Model` class. You can see more usage in `tests/test_model.py`.

### Provided sample tests
We have also provided some sample tests in `tests/` folder to help you become
familiar with the environment. They primarily serve as an example of how to
use the starter code API. You should definitely add more tests as you implement
more layers and models as you finish up the project.

To test out native C++ tests, simply do `make test` in your `build` folder.

To test out python-based tests, first install `pytest` then use `pytest`:

```
pip install pytest tensorflow
pytest test/
```

### A Note on Debugging
Because this project involves testing both C++ and its Python binding, debugging can
be a pain. Here is the recommended debugging strategy:

1. Write your C++ unit test first. You can either use `gdb` or simple print statement
to figure out which parts is wrong.
2. Once C++ is working, write out your Python tests. If you want to use `gdb` for
testing, you need to compile the Python binding with debugging symbol:

  ```
  $ DEBUG=1 pip install -e .
  ```

  Then gdb the python binary
  ```
  gdb python
  ```
  Then run your script in `gdb` with `run` command.

## Project Tasks
There are several task you need to accomplish in this project. Although they are
not required to be completed in order, it is highly recommended to do so as it
makes the debugging easier.

### Task 1: Train a CNN for CIFAR-10
Your model should achieve at least 60% classification accuracy on CIFAR-10.
Although you're free to use any framework you want, the start code is provided
with Tensorflow 2 in mind.

You will need weights for your own udnn model, since we are not going to
implement training in this project.

If you're unsure about how to set up a network to train or how to obtain
the dataset, keras' official website has an excellent
[code example](https://keras.io/examples/cifar10_cnn/) of setting
up basic layers and train them against CIFAR-10. All the layers used in the code
example are required in Task 2. Notice that you have to port that code
into Tensorflow2 keras interface, which is trivial to do.

### Task 2: Implement layers in C++
You will implement these layers in C++ using the given starter code. For each
layer, you are required to write a corresponding `pytest` unit test to test
the arithmetic against Tensorflow 2.

You are required to implement the following set of layers, in addition to the
layers you used in Task 1:
- Conv2D
- Dense
- Relu
- Sigmoid
- Flatten
- MaxPooling2D
- MinPooling2D [optional]
- DropOut

To add a custom layer to Python binding, you can look at the following code in
`python/udnn.cc`:
```C++
setup_layer<FlattenLayer, const TensorSize &>(m, "FlattenLayer");
```

You need to put the class type in the first template argument, followed
by its constructor signature. The string `"FlattenLayer` will be the class
name in Python. Then you should also add a helper function to
create a corresponding layer in `udnn/layer.py`. There are example code for you
to re-use.

You are required to write unit tests for each layer you implement, using
the same style as in `tests/test_layer.py`, i.e. constructing Tensorflow
layers with random input/weights and compare it against your own
implementation. Due to numeric precision error, you should use a delta
when comparing results.

### Task 3: Enhance C++ implementation with SIMD
You are required to re-implement all the layers with SIMD. We have
provided you with a nice SIMD library called
[xsimd](https://xsimd.readthedocs.io/en/latest/). The data in the `Tensor`
class is already memory aligned. You can read `tests/test_simd.cc` to see
how to use the SIMD library. In particular `TEST(simd, add)` demonstrates
idiomatic way to write SIMD arithmetic.

Keep in mind that not all layers can be implemented in SIMD easily, such as
Dropout layer.

Once you're done, you can use the unit tests you wrote from Task 2 to test
against your SIMD implementation. They should pass all the tests if your
SIMD implementation is correct.

Since xsimd is a portable library, you can develop your code on your own
Intel/AMD machine. However, before you continue to Task 4, make sure that
your code can be compiled and executed on an ARM CPU.

Some layers may require some "tricks" with SIMD implementation. For instance,
you can pre-compute the mask for drop out layer implementation with a
random number generator (this is what actually happens under the hood.)

### Task 4: Benchmark your SIMD implementation and further optimization
You should benchmark your SIMD version of udnn against Tensorflow2 on your
own machine as well as on your edge device, and use that to further optimize
your SIMD implementation. Keep in mind that arithmetic operations are only
a portion of time that the CPU spends executing the model, cache also plays
an important role. You can experiment around how to make your tensor storage
more cache friendly!

You should at least compare the following two cases with variable data types,
such as `int8`, `int16`, `int32`, `flaot32`, and `double`:
1. Benchmark performance impact on Intel (all data types in C++) on `Conv2D`
   and `Dense` layers.
2. Benchmark against Tensorflow on Intel CPU (`float32` and `double`) on model
   predication. This should be done in Python.
3. Benchmark performance impact on ARM (all data types) in C++ with SIMD
   and without SIMD.


Notice that Tensorflow CPU requires the Conv2D layer to be at least floats;
you only need to include `float32` and `double` on Intel CPU when comparing
with Tensorflow.


To make a fair comparison, make sure that you have enabled the release build
for C++. To do so, use the following cmake command inside your build folder:
```
cmake .. -DCMAKE_BUILD_TYPE=Release
```

We have included the example C++ benchmark code in `benchmark/benchmark.cc`
and Python benchmark code in `benchmark/benchmark.py`. Feel free
to build your own benchmark code on top of that.

### Extra credit:
Given the nature of this project, only the sky is the limit! Here is an
incomplete list of possible extra credits:
- Implement extra layers such as average pooling, concatenate, etc. Feel free
 to modify the starter code to do so.
- Implement quantization for your model and compare the performance.
- Out-perform Tensorflow implementation.
- Run inference on Arduino. The Arduino we have uses a 32-bit processor, which
allows up to 4 lanes of operations.
- ...

### Useful links:
- pybind11: https://pybind11.readthedocs.io/
- xsimd: https://xsimd.readthedocs.io/en/stable/index.html
- Tensorflow keras layers: https://www.tensorflow.org/api_docs/python/tf/keras/layers

## Submission
- Your source code without build artifacts in a zip file. Please make sure to
  populate the git submodules.
- Your write up on Task 4 in txt or pdf format.
