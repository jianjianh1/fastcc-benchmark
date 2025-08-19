#!/bin/bash

# ./build/benchmark ./fastcc_test_tensors/frostt/chicago-crime-comm.tns 1,2,3 1 1 | tee ./experiments/chicago123_1_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/chicago-crime-comm.tns 1,2,3 2 1 | tee ./experiments/chicago123_2_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/chicago-crime-comm.tns 1,2,3 4 1 | tee ./experiments/chicago123_4_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/chicago-crime-comm.tns 1,2,3 8 1 | tee ./experiments/chicago123_8_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/chicago-crime-comm.tns 1,2,3 16 1 | tee ./experiments/chicago123_16_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/chicago-crime-comm.tns 1,2,3 32 1 | tee ./experiments/chicago123_32_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/chicago-crime-comm.tns 1,2,3 64 1 | tee ./experiments/chicago123_64_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/chicago-crime-comm.tns 1,2,3 128 1 | tee ./experiments/chicago123_128_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/chicago-crime-comm.tns 1,2,3 256 1 | tee ./experiments/chicago123_256_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/chicago-crime-comm.tns 1,2,3 512 1 | tee ./experiments/chicago123_512_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/chicago-crime-comm.tns 1,2,3 1024 1 | tee ./experiments/chicago123_1024_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/chicago-crime-comm.tns 1,2,3 2048 1 | tee ./experiments/chicago123_2048_dense.txt
