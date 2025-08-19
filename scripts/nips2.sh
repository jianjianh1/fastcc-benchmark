# ./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 2 1 1 | tee ./experiments/nips2_1_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 2 2 1 | tee ./experiments/nips2_2_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 2 4 1 | tee ./experiments/nips2_4_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 2 8 1 | tee ./experiments/nips2_8_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 2 16 1 | tee ./experiments/nips2_16_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 2 32 1 | tee ./experiments/nips2_32_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 2 64 1 | tee ./experiments/nips2_64_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 2 128 1 | tee ./experiments/nips2_128_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 2 256 1 | tee ./experiments/nips2_256_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 2 512 1 | tee ./experiments/nips2_512_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 2 1024 1 | tee ./experiments/nips2_1024_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/nips.tns 2 2048 1 | tee ./experiments/nips2_2048_dense.txt
