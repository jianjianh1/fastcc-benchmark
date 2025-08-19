# ./build/benchmark ./fastcc_test_tensors/frostt/vast.tns 0,1 1 1 | tee ./experiments/vast01_1_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/vast.tns 0,1 2 1 | tee ./experiments/vast01_2_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/vast.tns 0,1 4 1 | tee ./experiments/vast01_4_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/vast.tns 0,1 8 1 | tee ./experiments/vast01_8_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/vast.tns 0,1 16 1 | tee ./experiments/vast01_16_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/vast.tns 0,1 32 1 | tee ./experiments/vast01_32_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/vast.tns 0,1 64 1 | tee ./experiments/vast01_64_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/vast.tns 0,1 128 1 | tee ./experiments/vast01_128_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/vast.tns 0,1 256 1 | tee ./experiments/vast01_256_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/vast.tns 0,1 512 1 | tee ./experiments/vast01_512_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/vast.tns 0,1 1024 1 | tee ./experiments/vast01_1024_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/vast.tns 0,1 2048 1 | tee ./experiments/vast01_2048_dense.txt
