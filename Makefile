csf_tensor: csf_tensor.cpp contract.hpp read.hpp
	g++ -std=c++20 -g -o csf_tensor csf_tensor.cpp

all: driver.cc contract.hpp read.hpp
	g++ -std=c++20 -Ofast -march=native -mtune=native -ffast-math -o driver driver.cc
	#g++ -std=c++20 -g -o driver driver.cc
