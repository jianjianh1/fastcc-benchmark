#include "contract.hpp"
#include <unistd.h>
#include "read.hpp"
#include <chrono>
#include <iostream>
#include <sys/resource.h>
#include <sys/wait.h>
#include <vector>

double make_a_run(Tensor<double> &some_tensor, std::string exp_name,
                CoOrdinate contr, int tile_size, bool dense) {
  int pipefd[2]; // pipefd[0] for reading, pipefd[1] for writing
  if (pipe(pipefd) == -1) {
    perror("pipe");
    return -1;
  }
  if (fork() == 0) {
    some_tensor._infer_dimensionality();
    some_tensor._infer_shape();
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    if (dense) {
      std::vector<long long> result =
          some_tensor.fastcc_multiply_timing<TileAccumulator<double>, double, double>(
              some_tensor, contr, contr, tile_size);
    } else {
      std::vector<long long> result =
          some_tensor
              .fastcc_multiply_timing<TileAccumulatorMap<double>, double, double>(
                  some_tensor, contr, contr, tile_size);
    }
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    close(pipefd[0]); // Close unused read end in child
    write(pipefd[1], &time_span, sizeof(time_span));
    close(pipefd[1]); // Close write end
    exit(0);
  } else {
    int stat;
    double received_value = -1.0;
    close(pipefd[1]); // Close unused write end in parent
    // Wait for the child process to terminate and get its exit status
    if (wait(&stat) == -1) {
      perror("waitpid");
      return -1.0;
    }
    if (WIFEXITED(stat)) {
      int exitStatus = WEXITSTATUS(stat);
      if (exitStatus == EXIT_SUCCESS) {
        // Child exited normally, try to read from the pipe
        if (read(pipefd[0], &received_value, sizeof(double)) == -1) {
          perror("parent read");
          exit(EXIT_FAILURE);
        }
        std::cout << "Parent received: " << received_value << std::endl;
      } else {
        std::cerr << "Parent: Child process failed with exit status "
                  << exitStatus << std::endl;
        return -1.0;
      }
    } else if (WIFSIGNALED(stat)) {
      std::cerr << "Parent: Child process terminated by signal "
                << WTERMSIG(stat) << std::endl;
      // Handle the child's termination by signal appropriately
      return -1.0;
    }
    close(pipefd[0]); // Close read end
    return received_value;
  }
}

double run_a_times_b(Tensor<double>& a, Tensor<double>& b, CoOrdinate contr, int tile_size, bool dense){
    int pipefd[2]; // pipefd[0] for reading, pipefd[1] for writing
    if (pipe(pipefd) == -1) {
        perror("pipe");
        return -1;
    }
    if (fork() == 0) {
        a._infer_dimensionality();
        a._infer_shape();
        b._infer_dimensionality();
        b._infer_shape();
        std::chrono::high_resolution_clock::time_point t1 =
            std::chrono::high_resolution_clock::now();
        if(dense){
            std::vector<long long> result = a.fastcc_multiply_timing<TileAccumulator<double>, double>(
                b, contr, contr, tile_size);
            auto start_time = std::chrono::high_resolution_clock::now();

            auto result_ = a.fastcc_multiply<TileAccumulator<double>, double>(
                b, contr, contr, tile_size);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
            std::cout << "Total time: " << compute_time << " microseconds" << std::endl;
        } else {
            std::vector<long long> result = a.fastcc_multiply_timing<TileAccumulatorMap<double>, double>(
                b, contr, contr, tile_size);
            auto start_time = std::chrono::high_resolution_clock::now();

            auto result_ = a.fastcc_multiply<TileAccumulatorMap<double>, double>(
                b, contr, contr, tile_size);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto compute_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
            std::cout << "Total time: " << compute_time << " microseconds" << std::endl;
        }
        std::chrono::high_resolution_clock::time_point t2 =
            std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_span =
            std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        close(pipefd[0]); // Close unused read end in child
        write(pipefd[1], &time_span, sizeof(time_span));
        close(pipefd[1]); // Close write end
        exit(0); // Exit the child process
    } else {
        int stat;
        double received_value = -1.0;
        close(pipefd[1]); // Close unused write end in parent
        // Wait for the child process to terminate and get its exit status
        if (wait(&stat) == -1) {
            perror("waitpid");
            return -1.0;
        }
        if (WIFEXITED(stat)) {
            int exitStatus = WEXITSTATUS(stat);
            if (exitStatus == EXIT_SUCCESS) {
                // Child exited normally, try to read from the pipe
                if (read(pipefd[0], &received_value, sizeof(double)) == -1) {
                    perror("parent read");
                    exit(EXIT_FAILURE);
                }
                std::cout << "Parent received: " << received_value << std::endl;
            } else {
                std::cerr << "Parent: Child process failed with exit status " << exitStatus << std::endl;
                return -1.0;
            }
        } else if (WIFSIGNALED(stat)) {
            std::cerr << "Parent: Child process terminated by signal " << WTERMSIG(stat) << std::endl;
            // Handle the child's termination by signal appropriately
            return -1.0;
        }
        close(pipefd[0]); // Close read end
        return received_value;
    }
}

void run_frostt_experiments(std::vector<int> tile_sizes, std::ostream& out, std::string& prefix) {

  double minimum_times[10] = {1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30};
  Tensor<double> nips = Tensor<double>(prefix + "/nips.tns", true);
  Tensor<double> chicago = Tensor<double>(prefix + "/chicago-crime-comm.tns", true);
  Tensor<double> vast = Tensor<double>(prefix + "/vast-2015-mc1-5d.tns", true);
  Tensor<double> uber = Tensor<double>(prefix + "/uber.tns", true);
  for (auto s : tile_sizes) {
        double time_taken = 0.0;
        // nips experiments
        std::cout << "Running nips tensor" << std::endl;
        time_taken = make_a_run(nips, "NIPS-2", CoOrdinate({2}), s * 1024, false);
        std::cout<<"Time taken for NIPS-2: " << time_taken << " seconds at tile size " << s << std::endl;
        minimum_times[0] = time_taken != -1.0 ? std::min(minimum_times[0], time_taken) : minimum_times[0];
        time_taken = make_a_run(nips, "NIPS-23", CoOrdinate({2, 3}), s * 1024, false);
        std::cout<<"Time taken for NIPS-23: " << time_taken << " seconds at tile size " << s << std::endl;
        minimum_times[1] = time_taken != -1.0 ? std::min(minimum_times[1], time_taken) : minimum_times[1];
        time_taken = make_a_run(nips, "NIPS-013", CoOrdinate({0, 1, 3}), s, true);
        std::cout<<"Time taken for NIPS-013: " << time_taken << " seconds at tile size " << s << std::endl;
        minimum_times[2] = time_taken != -1.0 ? std::min(minimum_times[2], time_taken) : minimum_times[2];

        ////////////// chicago experiments
        std::cout << "Running chicago tensor" << std::endl;
        time_taken = make_a_run(chicago, "Chicago-0", CoOrdinate({0}), s, true);
        std::cout<<"Time taken for Chicago-0: " << time_taken << " seconds at tile size " << s << std::endl;
        minimum_times[3] = time_taken != -1.0 ? std::min(minimum_times[3], time_taken) : minimum_times[3];
        time_taken = make_a_run(chicago, "Chicago-01", CoOrdinate({0, 1}), s, true);
        std::cout<<"Time taken for Chicago-01: " << time_taken << " seconds at tile size " << s << std::endl;
        minimum_times[4] = time_taken != -1.0 ? std::min(minimum_times[4], time_taken) : minimum_times[4];
        time_taken = make_a_run(chicago, "Chicago-123", CoOrdinate({1, 2, 3}), s, true);
        std::cout<<"Time taken for Chicago-123: " << time_taken << " seconds at tile size " << s << std::endl;
        minimum_times[5] = time_taken != -1.0 ? std::min(minimum_times[5], time_taken) : minimum_times[5];

        //////////////// vast-3d experiments
        std::cout << "Running vast-5d tensor" << std::endl;
        time_taken = make_a_run(vast, "Vast-5d-01", CoOrdinate({0, 1}), s, true);
        std::cout<<"Time taken for Vast-5d-01: " << time_taken << " seconds at tile size " << s << std::endl;
        minimum_times[6] = time_taken != -1.0 ? std::min(minimum_times[6], time_taken) : minimum_times[6];
        time_taken = make_a_run(vast, "Vast-5d-014", CoOrdinate({0, 1, 4}), s, true);
        std::cout<<"Time taken for Vast-5d-014: " << time_taken << " seconds at tile size " << s << std::endl;
        minimum_times[7] = time_taken != -1.0 ? std::min(minimum_times[7], time_taken) : minimum_times[7];

        /////////////////// uber experiments
        std::cout << "Running uber tensor" << std::endl;
        time_taken = make_a_run(uber, "Uber-02", CoOrdinate({0, 2}), s, true);
        std::cout<<"Time taken for Uber-02: " << time_taken << " seconds at tile size " << s << std::endl;
        minimum_times[8] = time_taken != -1.0 ? std::min(minimum_times[8], time_taken) : minimum_times[8];
        time_taken = make_a_run(uber, "Uber-123", CoOrdinate({1, 2, 3}), s, true);
        std::cout<<"Time taken for Uber-123: " << time_taken << " seconds at tile size " << s << std::endl;
        minimum_times[9] = time_taken != -1.0 ? std::min(minimum_times[9], time_taken) : minimum_times[9];
  }

  long l3_size_bytes = sysconf(_SC_LEVEL3_CACHE_SIZE);
  std::cout << "Going to run model, cache size is " << l3_size_bytes
            << " bytes, number of threads being used is "
            << std::thread::hardware_concurrency() << std::endl;

  auto get_sparse_size = [](double density) {
    long l3_size_bytes = sysconf(_SC_LEVEL3_CACHE_SIZE);
    return int(
        std::sqrt(l3_size_bytes /
                  (17.7 * std::thread::hardware_concurrency() * density)));
  };

  int dense_tile_size = int(std::sqrt(
      l3_size_bytes / (sizeof(double) * std::thread::hardware_concurrency())));
  int sparse_size_0 = get_sparse_size(4.6e-8);
  int sparse_size_1 = get_sparse_size(7.9e-7);
  std::cout << " Dense tile size for model: " << dense_tile_size << std::endl;
  std::cout << " Sparse tile size for model nips2: " << sparse_size_0 << std::endl;
  std::cout << " Sparse tile size for model nips23: " << sparse_size_1 << std::endl;

  double model_times[10] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  model_times[0] = make_a_run(nips, "NIPS-2", CoOrdinate({2}), sparse_size_0, false);
  model_times[1] = make_a_run(nips, "NIPS-23", CoOrdinate({2, 3}), sparse_size_1, false);
  model_times[2] = make_a_run(nips, "NIPS-013", CoOrdinate({0, 1, 3}), dense_tile_size, true);
  model_times[3] = make_a_run(chicago, "Chicago-0", CoOrdinate({0}), dense_tile_size, true);
  model_times[4] = make_a_run(chicago, "Chicago-01", CoOrdinate({0, 1}), dense_tile_size, true);
  model_times[5] = make_a_run(chicago, "Chicago-123", CoOrdinate({1, 2, 3}), dense_tile_size, true);
  model_times[6] = make_a_run(vast, "Vast-5d-01", CoOrdinate({0, 1}), dense_tile_size, true);
  model_times[7] = make_a_run(vast, "Vast-5d-014", CoOrdinate({0, 1, 4}), dense_tile_size, true);
  model_times[8] = make_a_run(uber, "Uber-02", CoOrdinate({0, 2}), dense_tile_size, true);
  model_times[9] = make_a_run(uber, "Uber-123", CoOrdinate({1, 2, 3}), dense_tile_size, true);

  out << "Chicago-0"
      << "," << minimum_times[3] << "," << model_times[3] << std::endl;
  out << "Chicago-01"
      << "," << minimum_times[4] << "," << model_times[4] << std::endl;
  out << "Chicago-123"
      << "," << minimum_times[5] << "," << model_times[5] << std::endl;
  out << "Vast-5d-01"
      << "," << minimum_times[6] << "," << model_times[6] << std::endl;
  out << "Vast-5d-014"
      << "," << minimum_times[7] << "," << model_times[7] << std::endl;
  out << "Uber-02"
      << "," << minimum_times[8] << "," << model_times[8] << std::endl;
  out << "Uber-123"
      << "," << minimum_times[9] << "," << model_times[9] << std::endl;
  out << "NIPS-2"
      << "," << minimum_times[0] << "," << model_times[0] << std::endl;
  out << "NIPS-23"
      << "," << minimum_times[1] << "," << model_times[1] << std::endl;
  out << "NIPS-013"
      << "," << minimum_times[2] << "," << model_times[2] << std::endl;
}

void run_chemistry_experiments(std::vector<int> tile_sizes, std::ostream &out,
                               std::string &caffeine_prefix,
                               std::string &guanine_prefix) {
  double minimum_times[6] = {1<<30, 1<<30, 1<<30, 1<<30, 1<<30, 1<<30};

  Tensor<double> tevv_caffeine =
      Tensor<double>(caffeine_prefix + "/TEvv.tns", true);
  Tensor<double> teoo_caffeine =
      Tensor<double>(caffeine_prefix + "/TEoo.tns", true);
  Tensor<double> teov_caffeine =
      Tensor<double>(caffeine_prefix + "/TEov.tns", true);
  Tensor<double> tevv_guanine =
      Tensor<double>(guanine_prefix + "/TEvv.tns", true);
  Tensor<double> teoo_guanine =
      Tensor<double>(guanine_prefix + "/TEoo.tns", true);
  Tensor<double> teov_guanine =
      Tensor<double>(guanine_prefix + "/TEov.tns", true);
  for (auto s : tile_sizes) {
        double time_taken = 0.0;
        ///////////// caffeine experiments
        std::cout << "Running caffeine-vvoo" << std::endl;
        time_taken = run_a_times_b(tevv_caffeine, teoo_caffeine,
                                   CoOrdinate({2}), s, true);
        std::cout << "Time taken for caffeine-vvoo: " << time_taken
                  << " seconds at tile size " << s << std::endl;
        minimum_times[0] = time_taken != -1.0
                               ? std::min(minimum_times[0], time_taken)
                               : minimum_times[0];
        time_taken = run_a_times_b(teov_caffeine, teov_caffeine,
                                   CoOrdinate({2}), s, true);
        std::cout << "Time taken for caffeine-ovov: " << time_taken
                  << " seconds at tile size " << s << std::endl;
        minimum_times[1] = time_taken != -1.0
                               ? std::min(minimum_times[1], time_taken)
                               : minimum_times[1];
        time_taken = run_a_times_b(tevv_caffeine, teov_caffeine,
                                   CoOrdinate({2}), s, true);
        std::cout << "Time taken for caffeine-vvov: " << time_taken
                  << " seconds at tile size " << s << std::endl;
        minimum_times[2] = time_taken != -1.0
                               ? std::min(minimum_times[2], time_taken)
                               : minimum_times[2];

        std::cout << "Running guanine-vvoo" << std::endl;
        time_taken =
            run_a_times_b(tevv_guanine, teoo_guanine, CoOrdinate({2}), s, true);
        std::cout << "Time taken for guanine-vvoo: " << time_taken
                  << " seconds at tile size " << s << std::endl;
        minimum_times[3] = time_taken != -1.0
                               ? std::min(minimum_times[3], time_taken)
                               : minimum_times[3];
        time_taken =
            run_a_times_b(teov_guanine, teov_guanine, CoOrdinate({2}), s, true);
        std::cout << "Time taken for guanine-ovov: " << time_taken
                  << " seconds at tile size " << s << std::endl;
        minimum_times[4] = time_taken != -1.0
                               ? std::min(minimum_times[4], time_taken)
                               : minimum_times[4];
        time_taken =
            run_a_times_b(tevv_guanine, teov_guanine, CoOrdinate({2}), s, true);
        std::cout << "Time taken for guanine-vvov: " << time_taken
                  << " seconds at tile size " << s << std::endl;
        minimum_times[5] = time_taken != -1.0
                               ? std::min(minimum_times[5], time_taken)
                               : minimum_times[5];
  }
}

int main(int argc, char** argv) {
    std::vector<int> grid_sizes = {64, 128, 256, 512, 1024, 2048};
    std::string caffeine_dir = "fastcc_test_tensors/caffeine/", guanine_dir = "fastcc_test_tensors/guanine/";
    std::ofstream results_chem;
    results_chem.open("chemistry_times.csv");
    results_chem << "tensor, time" << std::endl;
    run_chemistry_experiments(grid_sizes, results_chem, caffeine_dir, guanine_dir);
    results_chem.close();

    return 0;
}
