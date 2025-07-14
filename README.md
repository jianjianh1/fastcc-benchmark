This repository contains fast (CPU) kernels for sparse tensor times sparse tensor product.

1. Counting the number of FLOP(s) in sparse N-D tensor contractions with batch indices.
2. Computing the positions of non-zeros in the result of an N-D sparse tensor contraction.
3. Computing the output (including the data) of an N-D sparse tensor contraction.
Python bindings for the above three functions are exported using pybind11, in the file `pybind_wrapper.cpp`.

# Testing
The `driver.cc` file contains several unit-tests for the above functions.
It can be compiled using `make all`.

# Exporting Python bindings
Get taskflow here https://github.com/taskflow/taskflow
Modify the `tasks.py` file at lines 4 and 10 to point to the path to taskflow on your system.
Follow the steps here: https://realpython.com/python-bindings-overview/#pybind11
1. Install pybind11 using pip: `pip install pybind11`
2. Run `invoke`, and check to see that a shared object with the name `sparse_opcnt` is generated.

