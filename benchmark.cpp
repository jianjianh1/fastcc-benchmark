#include <iostream>
#include <string>
#include <chrono>

#include "contract.hpp"
#include "read.hpp"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <tensor_file> <mode: i_1,i_2,...> [tile_size] [dense: 0|1]" << std::endl;
        return 1;
    }
    std::string tensor_file = argv[1];
    std::string mode = argv[2];
    int tile_size = 128;
    if (argc > 3) {
        tile_size = std::stoi(argv[3]);
    }
    bool dense = false;
    if (argc > 4) {
        dense = std::stoi(argv[4]) != 0;
    }

    std::cout << "Loading tensor..." << std::endl;
    Tensor<double> a(tensor_file, true);
    std::cout << "Loaded tensor." << std::endl;

    // Choose contraction mode
    // Parse contraction modes from comma-separated string
    std::vector<int> contr_indices;
    std::string contr_str = argv[2];
    size_t start = 0, end = 0;
    while ((end = contr_str.find(',', start)) != std::string::npos) {
        contr_indices.push_back(std::stoi(contr_str.substr(start, end - start)));
        start = end + 1;
    }
    if (start < contr_str.size()) {
        contr_indices.push_back(std::stoi(contr_str.substr(start)));
    }
    CoOrdinate contr(contr_indices);

    // Benchmark function
    auto make_a_run = [](Tensor<double> &some_tensor, std::string exp_name,
                        CoOrdinate contr, int tile_size, bool dense) -> double {
        std::cout << "Experiment: " << exp_name << std::endl;
        some_tensor._infer_dimensionality();
        some_tensor._infer_shape();
        auto t1 = std::chrono::high_resolution_clock::now();
        if (dense) {
            ListTensor<double> result =
                some_tensor.fastcc_multiply<TileAccumulator<double>, double, double>(
                    some_tensor, contr, contr, tile_size);
            std::cout << "Result NNZ count: " << result.compute_nnz_count() << std::endl;
        } else {
            ListTensor<double> result =
                some_tensor.fastcc_multiply<TileAccumulatorMap<double>, double, double>(
                    some_tensor, contr, contr, tile_size);
            std::cout << "Result NNZ count: " << result.compute_nnz_count() << std::endl;
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span =
            std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        std::cout << "Elapsed: " << time_span.count() << " seconds" << std::endl;
        return time_span.count();
    };

    // Run benchmark
    double elapsed = make_a_run(a, tensor_file + mode + (dense ? "_dense" : "_sparse"), contr, tile_size, dense);
    std::cout << "Total time: " << elapsed << " seconds" << std::endl;
    return 0;
}