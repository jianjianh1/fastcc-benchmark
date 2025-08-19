# ./build/benchmark ./fastcc_test_tensors/frostt/uber.tns 0,2 1 1 | tee ./experiments/uber02_1_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/uber.tns 0,2 2 1 | tee ./experiments/uber02_2_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/uber.tns 0,2 4 1 | tee ./experiments/uber02_4_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/uber.tns 0,2 8 1 | tee ./experiments/uber02_8_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/uber.tns 0,2 16 1 | tee ./experiments/uber02_16_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/uber.tns 0,2 32 1 | tee ./experiments/uber02_32_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/uber.tns 0,2 64 1 | tee ./experiments/uber02_64_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/uber.tns 0,2 128 1 | tee ./experiments/uber02_128_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/uber.tns 0,2 256 1 | tee ./experiments/uber02_256_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/uber.tns 0,2 512 1 | tee ./experiments/uber02_512_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/uber.tns 0,2 1024 1 | tee ./experiments/uber02_1024_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/uber.tns 0,2 2048 1 | tee ./experiments/uber02_2048_dense.txt
