# ./build/benchmark ./fastcc_test_tensors/frostt/uber.tns 1,2,3 1 1 | tee ./experiments/uber123_1_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/uber.tns 1,2,3 2 1 | tee ./experiments/uber123_2_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/uber.tns 1,2,3 4 1 | tee ./experiments/uber123_4_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/uber.tns 1,2,3 8 1 | tee ./experiments/uber123_8_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/uber.tns 1,2,3 16 1 | tee ./experiments/uber123_16_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/uber.tns 1,2,3 32 1 | tee ./experiments/uber123_32_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/uber.tns 1,2,3 64 1 | tee ./experiments/uber123_64_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/uber.tns 1,2,3 128 1 | tee ./experiments/uber123_128_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/uber.tns 1,2,3 256 1 | tee ./experiments/uber123_256_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/uber.tns 1,2,3 512 1 | tee ./experiments/uber123_512_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/uber.tns 1,2,3 1024 1 | tee ./experiments/uber123_1024_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/uber.tns 1,2,3 2048 1 | tee ./experiments/uber123_2048_dense.txt
