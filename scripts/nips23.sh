# ./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 2,3 1 1 | tee ./experiments/nips23_1_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 2,3 2 1 | tee ./experiments/nips23_2_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 2,3 4 1 | tee ./experiments/nips23_4_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 2,3 8 1 | tee ./experiments/nips23_8_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 2,3 16 1 | tee ./experiments/nips23_16_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 2,3 32 1 | tee ./experiments/nips23_32_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 2,3 64 1 | tee ./experiments/nips23_64_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 2,3 128 1 | tee ./experiments/nips23_128_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 2,3 256 1 | tee ./experiments/nips23_256_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 2,3 512 1 | tee ./experiments/nips23_512_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 2,3 1024 1 | tee ./experiments/nips23_1024_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 2,3 2048 1 | tee ./experiments/nips23_2048_dense.txt
