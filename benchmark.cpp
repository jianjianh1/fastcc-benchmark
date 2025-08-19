#include <iostream>
#include <string>
#include <chrono>
#include <algorithm>

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
                        CoOrdinate contr, int tile_size, bool dense, int iters) -> std::vector<long long> {
        std::cout << "Experiment: " << exp_name << std::endl;
        some_tensor._infer_dimensionality();
        some_tensor._infer_shape();
        
        std::vector<long long> hash_create_times;
        std::vector<long long> compute_times;
        std::vector<long long> accumulate_times;
        std::vector<long long> drain_times;
        std::vector<long long> indexing_times;
        std::vector<long long> total_times;

        if (dense) {
            for (int i = 0; i < iters; i++) {
                auto timings =
                    some_tensor.fastcc_multiply_timing<TileAccumulator<double>, double, double>(
                        some_tensor, contr, contr, tile_size);
                hash_create_times.push_back(timings[0]);
                compute_times.push_back(timings[1]);
                accumulate_times.push_back(timings[2]);
                drain_times.push_back(timings[3]);
                indexing_times.push_back(timings[4]);

                auto start = std::chrono::high_resolution_clock::now();
                auto result = some_tensor.fastcc_multiply<TileAccumulator<double>, double, double>(
                    some_tensor, contr, contr, tile_size);
                auto end = std::chrono::high_resolution_clock::now();
                long long total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                total_times.push_back(total_time);
            }
        } else {
            for (int i = 0; i < iters; i++) {
                auto timings =
                    some_tensor.fastcc_multiply_timing<TileAccumulatorMap<double>, double, double>(
                        some_tensor, contr, contr, tile_size);
                hash_create_times.push_back(timings[0]);
                compute_times.push_back(timings[1]);
                accumulate_times.push_back(timings[2]);
                drain_times.push_back(timings[3]);
                indexing_times.push_back(timings[4]);

                auto start = std::chrono::high_resolution_clock::now();
                auto result = some_tensor.fastcc_multiply<TileAccumulatorMap<double>, double, double>(
                    some_tensor, contr, contr, tile_size);
                auto end = std::chrono::high_resolution_clock::now();
                long long total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                total_times.push_back(total_time);
            }
        }

        std::nth_element(hash_create_times.begin(), hash_create_times.begin() + hash_create_times.size() / 2, hash_create_times.end());
        std::nth_element(compute_times.begin(), compute_times.begin() + compute_times.size() / 2, compute_times.end());
        std::nth_element(accumulate_times.begin(), accumulate_times.begin() + accumulate_times.size() / 2, accumulate_times.end());
        std::nth_element(drain_times.begin(), drain_times.begin() + drain_times.size() / 2, drain_times.end());
        std::nth_element(indexing_times.begin(), indexing_times.begin() + indexing_times.size() / 2, indexing_times.end());
        std::nth_element(total_times.begin(), total_times.begin() + total_times.size() / 2, total_times.end());

        std::vector<long long> median_times = {
            hash_create_times[hash_create_times.size() / 2],
            compute_times[compute_times.size() / 2],
            accumulate_times[accumulate_times.size() / 2],
            drain_times[drain_times.size() / 2],
            indexing_times[indexing_times.size() / 2],
            total_times[total_times.size() / 2]
        };

        return median_times;
    };

    // Run the benchmark
    std::vector<long long> median_times = make_a_run(a, tensor_file + "_" + mode + "_" + std::to_string(tile_size) + (dense ? "_dense" : "_sparse"), contr, tile_size, dense, 3);
    std::cout << "Hash create time: " << median_times[0] << " us" << std::endl;
    std::cout << "Compute time: " << median_times[1] << " us" << std::endl;
    std::cout << "Accumulate time: " << median_times[2] << " us" << std::endl;
    std::cout << "Drain time: " << median_times[3] << " us" << std::endl;
    std::cout << "Indexing time: " << median_times[4] << " us" << std::endl;
    std::cout << "Total time: " << median_times[5] << " us" << std::endl;

    return 0;
}