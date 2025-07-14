INTEL_PATH=/opt/intel/oneapi/mkl/latest
LD_FLAGS=${INTEL_PATH}/lib/intel64
INTEL_INCLUDE=-I/opt/intel/oneapi/mpi/2021.9.0/include
LIB = -lmkl_rt

test: tests.cc contract.hpp read.hpp
	g++ -std=c++20 $(INTEL_INCLUDE) -L$(LD_FLAGS) -O3 -g -o tests tests.cc $(LIB)
