# ./build/benchmark ./fastcc_test_tensors/frostt/vast.tns 0,1,4 1 1 | tee ./experiments/vast014_1_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/vast.tns 0,1,4 2 1 | tee ./experiments/vast014_2_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/vast.tns 0,1,4 4 1 | tee ./experiments/vast014_4_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/vast.tns 0,1,4 8 1 | tee ./experiments/vast014_8_dense.txt
# ./build/benchmark ./fastcc_test_tensors/frostt/vast.tns 0,1,4 16 1 | tee ./experiments/vast014_16_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/vast.tns 0,1,4 32 1 | tee ./experiments/vast014_32_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/vast.tns 0,1,4 64 1 | tee ./experiments/vast014_64_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/vast.tns 0,1,4 128 1 | tee ./experiments/vast014_128_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/vast.tns 0,1,4 256 1 | tee ./experiments/vast014_256_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/vast.tns 0,1,4 512 1 | tee ./experiments/vast014_512_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/vast.tns 0,1,4 1024 1 | tee ./experiments/vast014_1024_dense.txt
./build/benchmark ./fastcc_test_tensors/frostt/vast.tns 0,1,4 2048 1 | tee ./experiments/vast014_2048_dense.txt
