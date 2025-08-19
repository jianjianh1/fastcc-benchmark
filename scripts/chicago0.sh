# ./build/benchmark ./fastcc_test_tensors/frostt/chicago-crime-comm.tns 0 1 1 | tee ./experiments/chicago0_1_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/chicago-crime-comm.tns 0 2 1 | tee ./experiments/chicago0_2_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/chicago-crime-comm.tns 0 4 1 | tee ./experiments/chicago0_4_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/chicago-crime-comm.tns 0 8 1 | tee ./experiments/chicago0_8_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/chicago-crime-comm.tns 0 16 1 | tee ./experiments/chicago0_16_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/chicago-crime-comm.tns 0 32 1 | tee ./experiments/chicago0_32_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/chicago-crime-comm.tns 0 64 1 | tee ./experiments/chicago0_64_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/chicago-crime-comm.tns 0 128 1 | tee ./experiments/chicago0_128_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/chicago-crime-comm.tns 0 256 1 | tee ./experiments/chicago0_256_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/chicago-crime-comm.tns 0 512 1 | tee ./experiments/chicago0_512_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/chicago-crime-comm.tns 0 1024 1 | tee ./experiments/chicago0_1024_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/chicago-crime-comm.tns 0 2048 1 | tee ./experiments/chicago0_2048_dense.txt