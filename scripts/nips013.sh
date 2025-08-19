#!/bin/bash

# ./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 0,1,3 1 1 | tee ./experiments/nips013_1_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 0,1,3 2 1 | tee ./experiments/nips013_2_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 0,1,3 4 1 | tee ./experiments/nips013_4_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 0,1,3 8 1 | tee ./experiments/nips013_8_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 0,1,3 16 1 | tee ./experiments/nips013_16_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 0,1,3 32 1 | tee ./experiments/nips013_32_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 0,1,3 64 1 | tee ./experiments/nips013_64_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 0,1,3 128 1 | tee ./experiments/nips013_128_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 0,1,3 256 1 | tee ./experiments/nips013_256_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 0,1,3 512 1 | tee ./experiments/nips013_512_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 0,1,3 1024 1 | tee ./experiments/nips013_1024_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 0,1,3 2048 1 | tee ./experiments/nips013_2048_dense.txt
