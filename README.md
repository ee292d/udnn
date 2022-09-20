
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
array into tensor. If we want to instantiate a `Tensor` with signed 8-bit byte
and shape `(4, 3, 2, 1)`, we can simply do
```Python
t = tensor((4, 3, 2, 1), "int8")
```

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

`udnn/model.py` allows you to construct a model in Python as well as load weights
from or dump weights to a folder. It mimics some interface of the `keras`
`Model` class. You can see more usage in `tests/test_model.py`.

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

## Assignment Tasks
There are several task you need to accomplish in this assignment. These tasks
require lots of debugging, so make sure to start early!

### Task 1: Implement layers in C++
You will implement these layers in C++ using the given starter code:
- Conv2D
- Dense
- Relu

In particular, you are required to implement the `forward()` functions and
`activate_function` function. You can check out how `MaxPoolingLayer` and
`FlattenLayer` are implemented.

Once your implementation is working, you can try out

```
pytest tests/test_layer.py tests/test_model.py -v
```

to see if your implementation matches with Tensorflow's. We will use this
test script to grade this part of the homework. Notice that in this part of
the assignment, performance is not a concern, as long as it is correct.

### Task 2: Enhance C++ implementation with SIMD
You are required to re-implement some layers with SIMD. We have
provided you with a nice SIMD library called
[xsimd](https://xsimd.readthedocs.io/en/latest/). The data in the `Tensor`
class is already memory aligned. You can read `tests/test_simd.cc` to see
how to use the SIMD library. In particular `TEST(simd, add)` demonstrates
idiomatic way to write SIMD arithmetic.

Once you're done, you can use `tests/test_layer_simd.py` to test
against your SIMD implementation. They should pass all the tests if your
SIMD implementation is correct.

Since xsimd is a portable library, you can develop your code on your own
Intel/AMD machine.

Layers required to use SIMD:
- Dense
- Conv2D

### Task 3: Quantization
You are required to re-implementation some layers with quantization.
We have provided you with another starter code `quantize.hh`.
This file contains many helper functions that help you to convert floating point
to fixed points, and to quantized numbers.

To understand how quantization works in the system, let's review how floating points
can be quantized to fixed point with bias and scaling factors. The major difference
between floating points and fixed points is that the decimal position remains the same
in fixed point numbers. An illustration for a 8-bit fixed point is shown below:

```
[ ][ ][ ][ ].[ ][ ][ ][ ]
 |          |
sign     decimal
```

Notice that we use the top bit as sign bit, 3 bits for the integer parts, and
4 bits for the fraction. The decimal position can be arbitrary, and we use the
middle position since it is easier to implement multiplication.

The arithmetics for fixed point straightforward. For addition and subtraction,
we can reuse the same integer arithmetics since the decimal position doesn't
change. For multiplication, however, the decimal position is shifted and we
need to take actions to prevent overflow, as shown below:


```
[ ][ ][ ][ ].[ ][ ][ ][ ] * [ ][ ][ ][ ].[ ][ ][ ][ ]
 |          |                |          |
sign     decimal            sign     decimal

=

[ ][ ][ ][ ][ ][ ][ ][ ].[ ][ ][ ][ ][ ][ ][ ][ ]
 |                      |
sign                 decimal
```

To convert it back to the 8-bit fixed point, we need to round the result and
clamp the output, as shown below:


```
[ ][ ][ ][ ]{[ ][ ][ ][ ].[ ][ ][ ][ ]}[ ][ ][ ][ ]
              |          |
             sign     decimal
```

Function `fix_mult` implements this multiplication algorithm. Notice it uses
truncation rounding mode, which can be less accurate than other rounding modes.
You are encouraged to implement better fixed point multiplication schemes.

Now that we have covered fixed point arithmetics, let's take a look at how quantization
works. Because the weights can form a much larger range than the fixed points, we
need to scale the numbers down to a much smaller range to prevent overflow. To do
so, we first define a zero point $a$, called bias. Then we scale everything
using a scaling factor $s$. Given any number $x$, the quantization equation is

$$ x' = \frac{x - a}{s}$$

And the un-quantization equation is

$$ x = x' * s + a $$

The arithmetics for scaled fixed points is a little more complex. For addition/subtraction,
we have

$$ x' + y' = \frac{x - a}{s} + \frac{y - a}{s} = \frac{x + y - 2a}{s} $$

Notice that the actual addition result is

$$ quantize(x + y) = \frac{x + y - a}{s} $$

We need to adjust the output of $x' + y'$ to get the correct result for $quantize(x + y)$.
The same logic applies to multiplication! You are encouraged to work out the math before
start to implement the code!

In `quantize.hh`, you will see some blank functions. In particular, `quantize`, `unquantize`,
`quantize_add` and `quantize_mult`. We recommended you implement these functions first and
test it out with `pytest tests/test_quantize.py -v` to make sure your implementation works.
Then you can work on `forward_quantized()` in `layer.hh`, which is straightforward.

### Task 4: Further optimization
I'm sure you've noticed that the implementation you have in Task 3 is very inefficient! We
can pre-compute many coefficients to avoid calculating the same value repeatedly. You should
complete `quantize_add_opt` and `quantize_mult_opt` to use these pre-computed coefficients,
and then use them in `forward_quantize()`. Feel free to adjust the function signatures as you
see fit.

Now that you have learned different ways to speed up computation, it is time to test it out!
You are free to make any SIMD and quantization optimization to make your code more performant.
The benchmark code takes arbitrary quantization bias and scale (in fixed point format). You
can adjust the number in `prepare_submission.py` to make the model more accurate. By default,
the bias is 0 and the scale is 4.

Here are some hints where you can optimize:

1. Although we're using quantized numbers, we are still computing values one at a time. If there
   is no overflow, there is no difference adding multiple fixed points at once, i.e. treat 4 8-bit
   fixed-points as a 32-bit integer and add them up. You can even apply SIMD to make the lanes
   wider!
2. Memory is another performance bottleneck. You can reduce the number of memory transpose by
   pre-transposing the weights!
3. You can run analysis on all activation values and determine the best scaling coefficients

The final scoring metrics is

$$ score = \frac{accuracy}{time} $$

where the time is measured wall clock time using CIFAR-10 test images. You can see more details
in `benchmark/benchmark.py`

### Useful links:
- pybind11: https://pybind11.readthedocs.io/
- xsimd: https://xsimd.readthedocs.io/en/stable/index.html
- Tensorflow keras layers: https://www.tensorflow.org/api_docs/python/tf/keras/layers
- Pete Warden's blog posts: https://petewarden.com/2016/05/03/how-to-quantize-neural-networks-with-tensorflow/

## Submission
Run the following command to get `submission.zip` for all tasks:

```
python benchmark/prepare_submission.py
```
