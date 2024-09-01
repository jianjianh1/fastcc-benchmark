csf_tensor: csf_tensor.cpp contract.hpp read.hpp
	g++ -std=c++20 -g -o csf_tensor csf_tensor.cpp

test: tests.cc contract.hpp read.hpp
	g++ -std=c++20 -I/home/saurabh/hopscotch-map/include -O3 -g -o tests tests.cc

all: driver.cc contract.hpp read.hpp
	g++ -std=c++20 -O3 -g -march=native -mtune=native -ffast-math -I/home/saurabh/taskflow -I/home/saurabh/hopscotch-map/include -o driver driver.cc
	#g++ -std=c++20 -O2 -g -I/home/saurabh/taskflow -o driver driver.cc
	#g++ -std=c++20 -g -I/home/saurabh/taskflow -o driver driver.cc
	#g++ -std=c++20 -fsanitize=address -g -I/home/saurabh/hopscotch-map/include -I/home/saurabh/taskflow -o driver driver.cc
