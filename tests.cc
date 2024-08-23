#include "contract.hpp"
#include "read.hpp"
#include "utils.hpp"
#include <chrono>
void dense_gemm_opcount() {
  // C(2, 2) = A(2, 5) * B(5, 2);
  int a_shape[2] = {2, 5};
  int b_shape[2] = {5, 2};
  Tensor<float> A(10);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 5; j++) {
      int coords[2] = {i, j};
      A.get_nonzeros().push_back(NNZ<float>(1.0, 2, coords));
    }
  }
  Tensor<float> B(10);
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 2; j++) {
      int coords[2] = {i, j};
      B.get_nonzeros().push_back(NNZ<float>(1.0, 2, coords));
    }
  }

  int num_mults = A.count_ops(B, CoOrdinate({1}), CoOrdinate({0}));
  FlopCounter<float> counter;
  for (int i = 0; i < 2; i++) {
    for (int k = 0; k < 2; k++) {
      int accum = 0;
      for (int j = 0; j < 5; j++) {
        int left[2] = {i, j};
        int right[2] = {j, k};
        auto leftcord = CoOrdinate(2, left);
        auto rightcord = CoOrdinate(2, right);

        accum += counter.mul(A[leftcord], B[rightcord]);
      }
    }
  }
  assert(counter.get_mults() == num_mults);
  std::cout << "Count using FlopCounter " << counter.get_mults() << std::endl;
  std::cout << "Count using HT " << num_mults << std::endl;
}

void dense_gemm_shape() {
  // C(2, 2) = A(2, 5) * B(5, 2);
  int a_shape[2] = {2, 5};
  int b_shape[2] = {5, 2};
  Tensor<float> A(10);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 5; j++) {
      int coords[2] = {i, j};
      A.get_nonzeros().push_back(NNZ<float>(1.0, 2, coords));
    }
  }
  Tensor<float> B(10);
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 2; j++) {
      int coords[2] = {i, j};
      B.get_nonzeros().push_back(NNZ<float>(1.0, 2, coords));
    }
  }

  auto output_coordinates = A.output_shape(B, CoOrdinate({1}), CoOrdinate({}),
                                           CoOrdinate({0}), CoOrdinate({}));
  std::unordered_set<CoOrdinate> ground_truth;
  for (int i = 0; i < 2; i++) {
    for (int k = 0; k < 2; k++) {
      int temparr[2] = {i, k};
      auto this_coord = CoOrdinate(2, temparr);
      ground_truth.insert(this_coord);
    }
  }
  std::vector<CoOrdinate> difference;
  for (auto &coord : output_coordinates) {
    std::cout << coord.get_index(0) << " " << coord.get_index(1) << std::endl;
    if (ground_truth.find(coord) == ground_truth.end()) {
      difference.push_back(coord);
    }
  }
  assert(difference.size() == 0);
}

void sparse_gemm_count() {
  // C(2, 2) = A(2, 5) * B(5, 2);
  int a_shape[2] = {2, 5};
  int b_shape[2] = {5, 2};
  Tensor<float> A(10);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 5; j++) {
      if (j % 2 != i)
        continue;
      int coords[2] = {i, j};
      A.get_nonzeros().push_back(NNZ<float>(1.0, 2, coords));
    }
  }
  Tensor<float> B(10);
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 2; j++) {
      if (j % 2 == i)
        continue;
      int coords[2] = {i, j};
      B.get_nonzeros().push_back(NNZ<float>(1.0, 2, coords));
    }
  }
  int num_mults = A.count_ops(B, CoOrdinate({1}), CoOrdinate({0}));
  std::cout << "Counting using hashmap done" << std::endl;
  FlopCounter<float> counter;
  for (int i = 0; i < 2; i++) {
    for (int k = 0; k < 2; k++) {
      int accum = 0;
      for (int j = 0; j < 5; j++) {
        int left[2] = {i, j};
        int right[2] = {j, k};
        auto leftcord = CoOrdinate(2, left);
        auto rightcord = CoOrdinate(2, right);
        float left_val = A[leftcord];
        float right_val = B[rightcord];
        if (left_val != 0 && right_val != 0)
          accum += counter.mul(left_val, right_val);
      }
    }
  }
  assert(counter.get_mults() == num_mults);
  std::cout << "Count using FlopCounter " << counter.get_mults() << std::endl;
  std::cout << "Count using HT " << num_mults << std::endl;
}

void sparse_gemm_shape() {
  // C(2, 2) = A(2, 5) * B(5, 2);
  int a_shape[2] = {2, 5};
  int b_shape[2] = {5, 2};
  Tensor<float> A(10);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 5; j++) {
      if (j % 2 != i)
        continue;
      int coords[2] = {i, j};
      A.get_nonzeros().push_back(NNZ<float>(1.0, 2, coords));
    }
  }
  Tensor<float> B(10);
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 2; j++) {
      if (j % 2 == i)
        continue;
      int coords[2] = {i, j};
      B.get_nonzeros().push_back(NNZ<float>(1.0, 2, coords));
    }
  }
  auto output_coordinates = A.output_shape(B, CoOrdinate({1}), CoOrdinate({}),
                                           CoOrdinate({0}), CoOrdinate({}));
  std::unordered_set<CoOrdinate> ground_truth;
  for (int i = 0; i < 2; i++) {
    for (int k = 0; k < 2; k++) {
      int temparr[2] = {i, k};
      auto this_coord = CoOrdinate(2, temparr);
      ground_truth.insert(this_coord);
    }
  }
  std::vector<CoOrdinate> difference;
  for (auto &coord : output_coordinates) {
    std::cout << coord.get_index(0) << " " << coord.get_index(1) << std::endl;
    if (ground_truth.find(coord) == ground_truth.end()) {
      difference.push_back(coord);
    }
  }
  assert(difference.size() == 0);
}

void fourd_tensor_contraction() {
  const int NNZ_COUNT = 5000;
  const int NUM_CONTR = 4;
  // I(7, 11, 9, 11) = T0(7, 11, 10, 12, 9) * T1(7, 11, 10, 12, 11);
  int a_shape[5] = {7, 11, 10, 12, 9};
  int b_shape[5] = {7, 11, 10, 12, 11};
  Tensor<float> A(NNZ_COUNT);
  int a_ctr = 0;
  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 11; j++) {
      for (int k = 0; k < 10; k++) {
        for (int l = 0; l < 12; l++) {
          for (int m = 0; m < 9; m++) {
            if (a_ctr == NNZ_COUNT)
              break;

            int coords[5] = {i, j, k, l, m};
            A.get_nonzeros().push_back(NNZ<float>(1.0, 5, coords));
            a_ctr++;
          }
        }
      }
    }
  }
  Tensor<float> B(NNZ_COUNT);
  int b_ctr = 0;
  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 11; j++) {
      for (int k = 0; k < 10; k++) {
        for (int l = 0; l < 12; l++) {
          for (int m = 0; m < 11; m++) {
            if (b_ctr == NNZ_COUNT)
              break;

            int coords[5] = {i, j, k, l, m};
            B.get_nonzeros().push_back(NNZ<float>(1.0, 5, coords));
            b_ctr++;
          }
        }
      }
    }
  }
  auto left_c = CoOrdinate({0, 1, 2, 3});
  auto right_c = CoOrdinate({0, 1, 2, 3});
  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();
  int num_mults = A.count_ops(B, left_c, right_c);
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  std::cout
      << "Time taken for count_ops "
      << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
      << std::endl;
  FlopCounter<float> counter;
  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 11; j++) {
      for (int k = 0; k < 9; k++) {
        for (int l = 0; l < 11; l++) {
          int accum = 0;
          for (int m = 0; m < 10; m++) {
            for (int n = 0; n < 12; n++) {
              int left[5] = {i, j, m, n, k};
              int right[5] = {i, j, m, n, l};
              auto leftcord = CoOrdinate(5, left);
              auto rightcord = CoOrdinate(5, right);
              float left_val = A[leftcord];
              float right_val = B[rightcord];
              if (left_val != 0 && right_val != 0)
                accum += counter.mul(left_val, right_val);
            }
          }
        }
      }
    }
  }
  std::cout << "Count using FlopCounter " << counter.get_mults() << std::endl;
  std::cout << "Count using HT " << num_mults << std::endl;
  assert(counter.get_mults() == num_mults);
}

void fourd_tensor_contraction_shape() {
  const int NNZ_COUNT = 5000;
  // I(7, 11, 9, 11) = T0(7, 11, 10, 12, 9) * T1(7, 11, 10, 12, 11);
  int a_shape[5] = {7, 11, 10, 12, 9};
  int b_shape[5] = {7, 11, 10, 12, 11};
  Tensor<float> A(NNZ_COUNT);
  int a_ctr = 0;
  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 11; j++) {
      for (int k = 0; k < 10; k++) {
        for (int l = 0; l < 12; l++) {
          for (int m = 0; m < 9; m++) {
            if (a_ctr == NNZ_COUNT)
              break;

            int coords[5] = {i, j, k, l, m};
            A.get_nonzeros().push_back(NNZ<float>(1.0, 5, coords));
            a_ctr++;
          }
        }
      }
    }
  }
  Tensor<float> B(NNZ_COUNT);
  int b_ctr = 0;
  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 11; j++) {
      for (int k = 0; k < 10; k++) {
        for (int l = 0; l < 12; l++) {
          for (int m = 0; m < 11; m++) {
            if (b_ctr == NNZ_COUNT)
              break;

            int coords[5] = {i, j, k, l, m};
            B.get_nonzeros().push_back(NNZ<float>(1.0, 5, coords));
            b_ctr++;
          }
        }
      }
    }
  }
  auto left_c = CoOrdinate({2, 3});
  auto left_batch = CoOrdinate({0, 1});
  auto right_c = CoOrdinate({2, 3});
  auto right_batch = CoOrdinate({0, 1});
  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();
  auto output_coordinates =
      A.output_shape(B, left_c, left_batch, right_c, right_batch);
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  std::cout
      << "Time taken for get_contraction_shape "
      << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
      << std::endl;

  std::unordered_set<CoOrdinate> ground_truth;
  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 11; j++) {
      for (int k = 0; k < 9; k++) {
        for (int l = 0; l < 11; l++) {
          for (int m = 0; m < 10; m++) {
            for (int n = 0; n < 12; n++) {
              int left[5] = {i, j, m, n, k};
              int right[5] = {i, j, m, n, l};
              auto leftcord = CoOrdinate(5, left);
              auto rightcord = CoOrdinate(5, right);
              float left_val = A[leftcord];
              float right_val = B[rightcord];
              if (left_val != 0 && right_val != 0)
                ground_truth.insert(CoOrdinate({i, j, k, l}));
            }
          }
        }
      }
    }
  }
  std::vector<CoOrdinate> outp_minus_ground;
  for (auto &coord : output_coordinates) {
    // std::cout << coord.get_index(0) << " " << coord.get_index(1) <<
    // std::endl;
    if (ground_truth.find(coord) == ground_truth.end()) {
      outp_minus_ground.push_back(coord);
    }
  }
  assert(outp_minus_ground.size() == 0);
  std::vector<CoOrdinate> ground_minus_outp;
  for (auto &coord : ground_truth) {
    if (output_coordinates.find(coord) == output_coordinates.end()) {
      ground_minus_outp.push_back(coord);
    }
  }
  assert(ground_minus_outp.size() == 0);
}

int main() {
    dense_gemm_opcount();
    std::cout << "Passed dense_gemm_opcount" << std::endl;
    dense_gemm_shape();
    std::cout << "Passed dense_gemm_shape" << std::endl;
    sparse_gemm_count();
    std::cout << "Passed sparse_gemm_count" << std::endl;
    sparse_gemm_shape();
    std::cout << "Passed sparse_gemm_shape" << std::endl;
    fourd_tensor_contraction();
    std::cout << "Passed fourd_tensor_contraction" << std::endl;
    fourd_tensor_contraction_shape();
    std::cout << "Passed fourd_tensor_contraction_shape" << std::endl;
    return 0;
}
