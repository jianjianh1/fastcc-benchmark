This repository contains fast (CPU) kernels for sparse tensor times sparse tensor product.


This is designed to be a header only repository. Please clone with `--recursive` to get the dependencies.
Once you have the code, just include "contract.hpp" and "read.hpp" headers to use the kernels.


To run `result(i, j, k) = a(i, c0, j, c1) * b(k, c0, c1)` in double precision, use the following code:

```
Tensor<double> a("a.tns");
Tensor<double> b("b.tns");
ListTensor<double> result = a.fastcc_multiply<TileAccumulator<double>, double>(b, {1, 3}, {1, 2});
```


See https://github.com/HPCRL/sparse_benchmark/blob/master/sc_ae_speedups.cc for more usage examples.
