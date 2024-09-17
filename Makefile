INTEL_PATH=/opt/intel/oneapi/mkl/latest
LD_FLAGS=${INTEL_PATH}/lib/intel64
INTEL_INCLUDE=-I/opt/intel/oneapi/mpi/2021.9.0/include
LIB = -lmkl_rt

csf_tensor: csf_tensor.cpp contract.hpp read.hpp
	g++ -std=c++20 -g -o csf_tensor csf_tensor.cpp

test: tests.cc contract.hpp read.hpp
	g++ -std=c++20 $(INTEL_INCLUDE) -I/home/saurabh/hopscotch-map/include -L$(LD_FLAGS) -O3 -g -o tests tests.cc $(LIB)

all: driver.cc contract.hpp read.hpp
	clang++ -std=c++20 -O3 -g -march=native -mtune=native -ffast-math -I/usr/include/c++/11 -I/usr/include/x86_64-linux-gnu/c++/11 $(INTEL_INCLUDE) -I/home/saurabh/taskflow -I/home/saurabh/hopscotch-map/include -L /usr/lib/gcc/x86_64-linux-gnu/11 -L$(LD_FLAGS) -o driver driver.cc $(LIB)
	#g++ -std=c++20 -O2 -g -I/home/saurabh/taskflow -o driver driver.cc
	#g++ -std=c++20 -g -I/home/saurabh/taskflow -o driver driver.cc
	#g++ -std=c++20 -fsanitize=address -g -I/home/saurabh/hopscotch-map/include -I/home/saurabh/taskflow -o driver driver.cc
