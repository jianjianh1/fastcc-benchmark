all: driver.cc contract.hpp
	g++ -std=c++11 -Ofast -o driver driver.cc
	#g++ -std=c++20 -g -o driver driver.cc
