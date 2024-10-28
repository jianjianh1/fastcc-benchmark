#include "contract.hpp"
#include <unordered_set>
#include "read.hpp"
#include "utils.hpp"
#include <chrono>
void dense_gemm_count() {
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
  A._infer_shape();
  B._infer_shape();

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
  A._infer_shape();
  B._infer_shape();

  auto output_coordinates = A.output_shape(B, CoOrdinate({1}), CoOrdinate({}),
                                           CoOrdinate({0}), CoOrdinate({}));
  std::vector<CoOrdinate> ground_truth;
  for (int i = 0; i < 2; i++) {
    for (int k = 0; k < 2; k++) {
      int temparr[2] = {i, k};
      auto this_coord = CoOrdinate(2, temparr);
      ground_truth.push_back(this_coord);
    }
  }
  std::vector<CoOrdinate> difference;
  for (auto &coord : output_coordinates) {
    std::cout << coord.get_index(0) << " " << coord.get_index(1) << std::endl;
    if (std::find(ground_truth.begin(), ground_truth.end(), coord) == ground_truth.end()) {
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
  A._infer_shape();
  B._infer_shape();
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
  A._infer_shape();
  B._infer_shape();
  auto output_coordinates = A.output_shape(B, CoOrdinate({1}), CoOrdinate({}),
                                           CoOrdinate({0}), CoOrdinate({}));
  std::vector<CoOrdinate> ground_truth;
  for (int i = 0; i < 2; i++) {
    for (int k = 0; k < 2; k++) {
      int temparr[2] = {i, k};
      auto this_coord = CoOrdinate(2, temparr);
      ground_truth.push_back(this_coord);
    }
  }
  std::vector<CoOrdinate> difference;
  for (auto &coord : output_coordinates) {
    std::cout << coord.get_index(0) << " " << coord.get_index(1) << std::endl;
    if (std::find(ground_truth.begin(), ground_truth.end(), coord) == ground_truth.end()) {
      difference.push_back(coord);
    }
  }
  assert(difference.size() == 0);
}

void fourd_tensor_contraction_count() {
  const int NNZ_COUNT = 1000;
  const int NUM_CONTR = 4;
  // I(7, 11, 9, 11) = T0(7, 11, 10, 8, 9) * T1(7, 11, 10, 8, 11);
  int a_shape[5] = {7, 11, 10, 8, 9};
  int b_shape[5] = {7, 11, 10, 8, 11};
  Tensor<float> A(NNZ_COUNT);
  int a_ctr = 0;
  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 11; j++) {
      for (int k = 0; k < 10; k++) {
        for (int l = 0; l < 8; l++) {
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
        for (int l = 0; l < 8; l++) {
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
  A._infer_shape();
  B._infer_shape();
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
            for (int n = 0; n < 8; n++) {
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
  assert(counter.get_mults() == 9000);
}

void fourd_tensor_contraction_shape() {
  const int NNZ_COUNT = 1000;
  // I(7, 11, 9, 11) = T0(7, 11, 10, 8, 9) * T1(7, 11, 10, 8, 11);
  int a_shape[5] = {7, 11, 10, 8, 9};
  int b_shape[5] = {7, 11, 10, 8, 11};
  Tensor<float> A(NNZ_COUNT);
  int a_ctr = 0;
  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 11; j++) {
      for (int k = 0; k < 10; k++) {
        for (int l = 0; l < 8; l++) {
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
        for (int l = 0; l < 8; l++) {
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
  A._infer_shape();
  B._infer_shape();
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

  std::vector<CoOrdinate> ground_truth;
  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 11; j++) {
      for (int k = 0; k < 9; k++) {
        for (int l = 0; l < 11; l++) {
          for (int m = 0; m < 10; m++) {
            for (int n = 0; n < 8; n++) {
              int left[5] = {i, j, m, n, k};
              int right[5] = {i, j, m, n, l};
              auto leftcord = CoOrdinate(5, left);
              auto rightcord = CoOrdinate(5, right);
              float left_val = A[leftcord];
              float right_val = B[rightcord];
              if (left_val != 0 && right_val != 0)
                ground_truth.push_back(CoOrdinate({i, j, k, l}));
            }
          }
        }
      }
    }
  }
  std::cout<<"Manually ran shape"<<std::endl;
  std::vector<CoOrdinate> outp_minus_ground;
  for (auto &coord : output_coordinates) {
    // std::cout << coord.get_index(0) << " " << coord.get_index(1) <<
    // std::endl;
    if (std::find(ground_truth.begin(), ground_truth.end(), coord) == ground_truth.end()) {
      outp_minus_ground.push_back(coord);
    }
  }
  assert(outp_minus_ground.size() == 0);
  std::cout<<"Precision 100%"<<std::endl;
  SymbolicTensor gt_tensor(ground_truth.begin(), ground_truth.end());
  gt_tensor.get_shape();
  std::vector<CoOrdinate> ground_minus_outp;
  for (auto &coord : gt_tensor) {
    if (output_coordinates.find(coord) == output_coordinates.end()) {
      ground_minus_outp.push_back(coord);
    }
  }
  assert(ground_minus_outp.size() == 0);
}

void dense_multiply() {
  // I(2, 3, 6, 7) = T0(2, 3, 4, 5, 6) * T1(2, 3, 4, 5, 7);
  Tensor<float> T0(520);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 4; k++) {
        for (int l = 0; l < 5; l++) {
          for (int m = 0; m < 6; m++) {
            float data = i * j * k * l * m;
            T0.get_nonzeros().push_back(
                NNZ<float>(data, CoOrdinate({i, j, k, l, m})));
          }
        }
      }
    }
  }

  Tensor<float> T1(840);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 4; k++) {
        for (int l = 0; l < 5; l++) {
          for (int m = 0; m < 7; m++) {
            float data = i * j * k * l * m;
            T1.get_nonzeros().push_back(
                NNZ<float>(data, CoOrdinate({i, j, k, l, m})));
          }
        }
      }
    }
  }

  Tensor<float> I =
      T0.multiply<float>(T1, CoOrdinate({2, 3}), CoOrdinate({0, 1}),
                         CoOrdinate({2, 3}), CoOrdinate({0, 1}));

  Tensor<float> I_inout =
      T0.inner_outer_multiply<float>(T1, CoOrdinate({2, 3}), CoOrdinate({0, 1}),
                         CoOrdinate({2, 3}), CoOrdinate({0, 1}));

  Tensor<float> ground_truth(520);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      for (int m = 0; m < 6; m++) {
        for (int n = 0; n < 7; n++) {
          float acc = 0.0;
          for (int k = 0; k < 4; k++) {
            for (int l = 0; l < 5; l++) {
              float left_data = i * j * k * l * m;
              float right_data = i * j * k * l * n;
              acc += left_data * right_data;
            }
          }
          int res_coords[4] = {i, j, m, n};
          ground_truth.get_nonzeros().push_back(NNZ<float>(acc, 4, res_coords));
        }
      }
    }
  }
  IndexedTensor<float> ground_truth_indexed(ground_truth,
                                            CoOrdinate({0, 1, 2, 3}));
  IndexedTensor<float> i_indexed(I, CoOrdinate({0, 1, 2, 3}));
  IndexedTensor<float> inout_indexed(I_inout, CoOrdinate({0, 1, 2, 3}));
  assert(i_indexed == ground_truth_indexed);
  std::cout << "precision 100%" << std::endl;
  assert(ground_truth_indexed == i_indexed);
  std::cout << "recall 100%" << std::endl;
  assert(inout_indexed == ground_truth_indexed);
  assert(ground_truth_indexed == inout_indexed);
  std::cout<<"inner outer matches ground truth"<<std::endl;
}

bool one_in_x() {
#define X 2
  static uint count;
  count++;
  return count % X == 0;
}

void sparse_multiply() {
  // I(2, 3, 6, 7) = T0(2, 3, 4, 5, 6) * T1(2, 3, 4, 5, 7);
  Tensor<float> T0(520);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 4; k++) {
        for (int l = 0; l < 5; l++) {
          for (int m = 0; m < 6; m++) {
            if (one_in_x()) {
              float data = (i * j * k * l * m) + 1.0;
              T0.get_nonzeros().push_back(
                  NNZ<float>(data, CoOrdinate({i, j, k, l, m})));
            }
          }
        }
      }
    }
  }

  Tensor<float> T1(840);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 4; k++) {
        for (int l = 0; l < 5; l++) {
          for (int m = 0; m < 7; m++) {
            if (one_in_x()) {
              float data = (i * j * k * l * m) + 1.0;
              T1.get_nonzeros().push_back(
                  NNZ<float>(data, CoOrdinate({i, j, k, l, m})));
            }
          }
        }
      }
    }
  }

  Tensor<float> I =
      T0.multiply<float>(T1, CoOrdinate({2, 3}), CoOrdinate({0, 1}),
                         CoOrdinate({2, 3}), CoOrdinate({0, 1}));
  Tensor<float> I_inout =
      T0.inner_outer_multiply<float>(T1, CoOrdinate({2, 3}), CoOrdinate({0, 1}),
                         CoOrdinate({2, 3}), CoOrdinate({0, 1}));

  Tensor<float> ground_truth(520);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      for (int m = 0; m < 6; m++) {
        for (int n = 0; n < 7; n++) {
          float acc = 0.0;
          for (int k = 0; k < 4; k++) {
            for (int l = 0; l < 5; l++) {
              float left_data = T0[CoOrdinate({i, j, k, l, m})];
              float right_data = T1[CoOrdinate({i, j, k, l, n})];
              acc += left_data * right_data;
            }
          }
          if (acc != 0) {
            int res_coords[4] = {i, j, m, n};
            ground_truth.get_nonzeros().push_back(
                NNZ<float>(acc, 4, res_coords));
          }
        }
      }
    }
  }
  assert(ground_truth.reduce() > 0);
  IndexedTensor<float> ground_truth_indexed(ground_truth,
                                            CoOrdinate({0, 1, 2, 3}));
  IndexedTensor<float> i_indexed(I, CoOrdinate({0, 1, 2, 3}));
  IndexedTensor<float> inout_indexed(I_inout, CoOrdinate({0, 1, 2, 3}));
  assert(i_indexed == ground_truth_indexed);
  std::cout << "precision 100%" << std::endl;
  assert(ground_truth_indexed == i_indexed);
  std::cout << "recall 100%" << std::endl;
  assert(inout_indexed == ground_truth_indexed);
  assert(ground_truth_indexed == inout_indexed);
  std::cout<< "Inner outer matches ground truth"<<std::endl;
}

void sparse_multiply_offsetcrd() {
  // I(2, 3, 6, 7) = T0(2, 3, 4, 5, 6) * T1(2, 3, 4, 5, 7);
  Tensor<float> T0(520);
  const int OFFSET = 5; // FAHVE
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 4; k++) {
        for (int l = 0; l < 5; l++) {
          for (int m = 0; m < 6; m++) {
            if (one_in_x()) {
              float data = (i * j * k * l * m) + 1.0;
              T0.get_nonzeros().push_back(NNZ<float>(
                  data, CoOrdinate({i + OFFSET, j + OFFSET, k + OFFSET,
                                    l + OFFSET, m + OFFSET})));
            }
          }
        }
      }
    }
  }

  Tensor<float> T1(840);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 4; k++) {
        for (int l = 0; l < 5; l++) {
          for (int m = 0; m < 7; m++) {
            if (one_in_x()) {
              float data = (i * j * k * l * m) + 1.0;
              T1.get_nonzeros().push_back(NNZ<float>(
                  data, CoOrdinate({i + OFFSET, j + OFFSET, k + OFFSET,
                                    l + OFFSET, m + OFFSET})));
            }
          }
        }
      }
    }
  }

  Tensor<float> I =
      T0.multiply<float>(T1, CoOrdinate({2, 3}), CoOrdinate({0, 1}),
                         CoOrdinate({2, 3}), CoOrdinate({0, 1}));

  Tensor<float> I_inout =
      T0.inner_outer_multiply<float>(T1, CoOrdinate({2, 3}), CoOrdinate({0, 1}),
                         CoOrdinate({2, 3}), CoOrdinate({0, 1}));

  Tensor<float> ground_truth(520);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 3; j++) {
      for (int m = 0; m < 6; m++) {
        for (int n = 0; n < 7; n++) {
          float acc = 0.0;
          for (int k = 0; k < 4; k++) {
            for (int l = 0; l < 5; l++) {
              float left_data =
                  T0[CoOrdinate({i + OFFSET, j + OFFSET, k + OFFSET, l + OFFSET,
                                 m + OFFSET})];
              float right_data =
                  T1[CoOrdinate({i + OFFSET, j + OFFSET, k + OFFSET, l + OFFSET,
                                 n + OFFSET})];
              acc += left_data * right_data;
            }
          }
          if (acc != 0) {
            int res_coords[4] = {i + OFFSET, j + OFFSET, m + OFFSET,
                                 n + OFFSET};
            ground_truth.get_nonzeros().push_back(
                NNZ<float>(acc, 4, res_coords));
          }
        }
      }
    }
  }
  assert(ground_truth.reduce() > 0);
  IndexedTensor<float> ground_truth_indexed(ground_truth,
                                            CoOrdinate({0, 1, 2, 3}));
  IndexedTensor<float> i_indexed(I, CoOrdinate({0, 1, 2, 3}));

  IndexedTensor<float> inout_indexed(I_inout, CoOrdinate({0, 1, 2, 3}));
  assert(i_indexed == ground_truth_indexed);
  std::cout << "precision 100%" << std::endl;
  assert(ground_truth_indexed == i_indexed);
  std::cout << "recall 100%" << std::endl;
  assert(inout_indexed == ground_truth_indexed);
  assert(ground_truth_indexed == inout_indexed);
  std::cout<<"Inner outer matches ground truth"<<std::endl;
}

void sparse_multiply_extrnonly() {
  // I(6, 7) = T0(6, 4, 5) * T1(4, 5, 7);
  Tensor<float> T0(520);
  const int OFFSET = 5; // FAHVE
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 5; k++) {
        if (one_in_x()) {
          float data = (i * j * k) + 1.0;
          T0.get_nonzeros().push_back(NNZ<float>(
              data, CoOrdinate({i + OFFSET, j + OFFSET, k + OFFSET})));
        }
      }
    }
  }

  Tensor<float> T1(840);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 5; j++) {
      for (int k = 0; k < 7; k++) {
        if (one_in_x()) {
          float data = (i * j * k) + 1.0;
          T1.get_nonzeros().push_back(NNZ<float>(data, CoOrdinate({
                                                           i + OFFSET,
                                                           j + OFFSET,
                                                           k + OFFSET,
                                                       })));
        }
      }
    }
  }
  T0._infer_shape();
  T1._infer_shape();
  std::cout<<"T0 Shape was "<<T0.get_shape_string()<<std::endl;
  std::cout<<"T1 Shape was "<<T1.get_shape_string()<<std::endl;

  Tensor<float> I = T0.multiply<float>(T1, CoOrdinate({1, 2}), CoOrdinate({}),
                                       CoOrdinate({0, 1}), CoOrdinate({}));

  Tensor<float> I_inout = T0.inner_outer_multiply<float>(T1, CoOrdinate({1, 2}), CoOrdinate({}),
                                       CoOrdinate({0, 1}), CoOrdinate({}));

  Tensor<float> ground_truth(520);
  for (int i = 0; i < 6; i++) {
    for (int j = 0; j < 7; j++) {
      float acc = 0.0;
      for (int k = 0; k < 4; k++) {
        for (int l = 0; l < 5; l++) {
          float left_data =
              T0[CoOrdinate({i + OFFSET, k + OFFSET, l + OFFSET})];
          float right_data =
              T1[CoOrdinate({k + OFFSET, l + OFFSET, j + OFFSET})];
          acc += left_data * right_data;
        }
      }
      if (acc != 0) {
        int res_coords[2] = {i + OFFSET, j + OFFSET};
        ground_truth.get_nonzeros().push_back(NNZ<float>(acc, 2, res_coords));
      }
    }
  }
  I._infer_shape();
  ground_truth._infer_shape();
  assert(ground_truth.reduce() > 0);
  assert(I.reduce() > 0);
  IndexedTensor<float> ground_truth_indexed(ground_truth,
                                            CoOrdinate({0, 1}));
  IndexedTensor<float> i_indexed(I, CoOrdinate({0, 1}));
  IndexedTensor<float> inout_indexed(I_inout, CoOrdinate({0, 1}));
  assert(i_indexed == ground_truth_indexed);
  std::cout << "precision 100%" << std::endl;
  assert(ground_truth_indexed == i_indexed);
  std::cout << "recall 100%" << std::endl;
  assert(inout_indexed == ground_truth_indexed);
  assert(ground_truth_indexed == inout_indexed);
  std::cout << "inner outer matches ground truth" << std::endl;
}

void teov_dlmop_opcount() {
  Tensor<double> dteov("./test_data/TEov.tns", true);
  Tensor<densevec> dlmop("./test_data/d_LMOP.tns", true);
  auto left_c = CoOrdinate(std::vector<int>({0, 1})); // batch,contraction
  auto right_c = CoOrdinate(std::vector<int>({1, 2})); // batch,contraction
  dteov._infer_shape();
  dlmop._infer_shape();
  auto count_ops0 = dteov.count_ops(dlmop, left_c, right_c);
  left_c = CoOrdinate(std::vector<int>({1, 0}));
  right_c = CoOrdinate(std::vector<int>({2, 1}));
  auto count_ops1 = dteov.count_ops(dlmop, left_c, right_c);
  assert(count_ops0 == count_ops1);
}

Tensor<densevec> tevv_dlmop_fillvalues(Tensor<double>& dtevv, Tensor<densevec>& dlmop) {
    // [b_mu, K, m, i, e_mi] = (TEvv[[b_mu, e_mu, K]] * d_LMOP[[m, i, e_mu,
    // e_mi]]) contract e_mu
    auto left_c = CoOrdinate(std::vector<int>({1}));
    auto left_b = CoOrdinate(std::vector<int>({}));
    auto right_c = CoOrdinate(std::vector<int>({2}));
    auto right_b = CoOrdinate(std::vector<int>({}));
    Tensor<densevec> result;
    auto start = std::chrono::high_resolution_clock::now();
    result.fill_values(dtevv, dlmop, left_c, left_b, right_c, right_b);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken for fillvalues "
              << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                       start)
                     .count()
              << " microseconds" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    Tensor<densevec> result2 = dtevv.multiply<densevec>(dlmop, left_c, left_b, right_c, right_b);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken for full outer kernel "
              << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                       start)
                     .count()
              << " microseconds" << std::endl;
    std::cout<<"Size of result2 is "<<result2.get_size()<<std::endl;
    start = std::chrono::high_resolution_clock::now();
    Tensor<densevec> result_inout = dtevv.inner_outer_multiply<densevec>(dlmop, left_c, left_b, right_c, right_b);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken for inner outer kernel "
              << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                       start)
                     .count()
              << " microseconds" << std::endl;
    std::cout<<"Size of result_inout is "<<result_inout.get_size()<<std::endl;
    return result;
}

SymbolicTensor tevv_dlmop_outputshape(Tensor<double>& dtevv, Tensor<densevec>& dlmop) {
    // [b_mu, K, m, i, e_mi] = (TEvv[[b_mu, e_mu, K]] * d_LMOP[[m, i, e_mu,
    // e_mi]]) contract e_mu
    auto left_c = CoOrdinate(std::vector<int>({1}));
    auto left_b = CoOrdinate(std::vector<int>({}));
    auto right_c = CoOrdinate(std::vector<int>({2}));
    auto right_b = CoOrdinate(std::vector<int>({}));
    auto left_symbolic = SymbolicTensor(dtevv);
    auto right_symbolic = SymbolicTensor(dlmop);
    auto res =
        left_symbolic.contract(right_symbolic, left_c, left_b, right_c, right_b);
    std::cout << "Time taken for get_contraction_shape "
              << res.second
              << " microseconds" << std::endl;
    return res.first;
}

void tevv_dlmop_outputshape(){
    Tensor<double> tevv("./test_data/TEvv.tns", true);
    Tensor<densevec> d_LMOP("./test_data/d_LMOP.tns", true);
    tevv._infer_shape();
    d_LMOP._infer_shape();
    SymbolicTensor output_shape = tevv_dlmop_outputshape(tevv, d_LMOP);
    Tensor<densevec> output_fv = tevv_dlmop_fillvalues(tevv, d_LMOP);
    assert(output_shape.get_size() == output_fv.get_size());
    for (auto &nnz : output_fv) {
        bool found = false;
        for (auto &coord : output_shape) {
          if (coord == nnz.get_coords()) {
            found = true;
            break;
          }
        }
        if (!found) {
          std::cout << "Something in fill values not found in Symbolic Tensor "
                    << nnz.get_coords().to_string() << ": "
                    << nnz.get_data().to_string() << std::endl;
        }
    }
    for (auto &coord : output_shape) {
        bool found = false;
        for (auto &nnz : output_fv) {
          if (coord == nnz.get_coords()) {
            found = true;
            break;
          }
        }
        if (!found)
          std::cout << "Something in Symbolic Tensor not found in full output "
                    << coord.to_string() << std::endl;
    }

}

void symtensor_inf_shape() {
    Tensor<double> teov("./test_data/TEov.tns", true);
    teov._infer_shape();
    SymbolicTensor symteov(teov);
    std::vector<int> shape = symteov.get_shape();
    std::vector<int> ground_truth = {4, 24, 375}; //1 indexed
    //std::vector<int> ground_truth = {3, 23, 374}; //0 indexed
    assert(shape == ground_truth);
}

void dense_opcount(){
    // res(i, k, j, e_ij) = teov(i, e_mu, k) * dlmop(i, j, e_mu, e_ij)
    // MO = 4
    // PAO = 24
    // AUX = 375
    Tensor<double> teov("./test_data/TEov.tns", true);
    Tensor<densevec> dlmop("./test_data/d_LMOP.tns", true);
    SymbolicTensor symteov(teov);
    SymbolicTensor symdlmop(dlmop);
    auto left_c = CoOrdinate(std::vector<int>({1}));
    auto left_b = CoOrdinate(std::vector<int>({0}));
    auto right_c = CoOrdinate(std::vector<int>({2}));
    auto right_b = CoOrdinate(std::vector<int>({0}));
    auto res_pair = symteov.contract_dense(symdlmop, left_c, left_b, right_c, right_b);
    assert(res_pair.second == 4 * 4 * 24 * 375);
    SymbolicTensor res = res_pair.first;
    std::vector<int> shape = res.get_shape();
    std::vector<int> ground_truth = {4, 375, 4};
    assert(shape == ground_truth);
    // res2(i, f_mu, j, e_ij) = teov(i, f_mu, k) * res(i, k, j, e_ij)
    left_c = CoOrdinate(std::vector<int>({2}));
    left_b = CoOrdinate(std::vector<int>({0}));
    right_c = CoOrdinate(std::vector<int>({1}));
    right_b = CoOrdinate(std::vector<int>({0}));
    auto res2_pair = symteov.contract_dense(res, left_c, left_b, right_c, right_b);
    assert(res2_pair.second == 4 * 375 * 4 * 24);
    SymbolicTensor res2 = res2_pair.first;
    shape = res2.get_shape();
    ground_truth = {4, 24, 4};
    assert(shape == ground_truth);
}

//void fill_values_only_nobatch_scalar() {
//    // res(i, e_mu, j, f_mu) = teov(i, e_mu, k) * teov(j, f_mu, k)
//    Tensor<double> teov("./test_data/TEov.tns", true);
//    auto left_c = CoOrdinate(std::vector<int>({2}));
//    auto left_b = CoOrdinate(std::vector<int>({}));
//    auto right_c = CoOrdinate(std::vector<int>({2}));
//    auto right_b = CoOrdinate(std::vector<int>({}));
//    std::chrono::high_resolution_clock::time_point t1 =
//        std::chrono::high_resolution_clock::now();
//    Tensor<double> res =
//        teov.multiply<double>(teov, left_c, left_b, right_c, right_b);
//    std::chrono::high_resolution_clock::time_point t2 =
//        std::chrono::high_resolution_clock::now();
//    std::cout << "Time taken for multiply "
//              << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
//                     .count()
//              << std::endl;
//    Tensor<double> res2 =
//        teov.multiply<double>(teov, left_c, left_b, right_c, right_b);
//    res2._infer_dimensionality();
//    t1 = std::chrono::high_resolution_clock::now();
//    res2._fill_values_only(teov, teov, left_c, left_b, right_c, right_b);
//    t2 = std::chrono::high_resolution_clock::now();
//    std::cout << "Time taken for fill_values_only doubles "
//              << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
//                     .count()
//              << std::endl;
//    CoOrdinate result_idx({});
//    res._infer_dimensionality();
//    result_idx.all_positions(res.get_dimensionality());
//    assert(IndexedTensor<double>(res, result_idx) ==
//           IndexedTensor<double>(res2, result_idx));
//    assert(IndexedTensor<double>(res2, result_idx) ==
//           IndexedTensor<double>(res, result_idx));
//}

//void fill_values_only() {
//    // res(i, k, j, e_ij) = teov(i, e_mu, k) * dlmop(i, j, e_mu, e_ij)
//    Tensor<float> teov("./test_data/TEov.tns", true);
//    Tensor<densevec> dlmop("./test_data/d_LMOP.tns", true);
//    auto left_c = CoOrdinate(std::vector<int>({1}));
//    auto left_b = CoOrdinate(std::vector<int>({0}));
//    auto right_c = CoOrdinate(std::vector<int>({2}));
//    auto right_b = CoOrdinate(std::vector<int>({0}));
//    std::chrono::high_resolution_clock::time_point t1 =
//        std::chrono::high_resolution_clock::now();
//    Tensor<densevec> res =
//        teov.multiply<densevec>(dlmop, left_c, left_b, right_c, right_b);
//    std::chrono::high_resolution_clock::time_point t2 =
//        std::chrono::high_resolution_clock::now();
//    std::cout << "Time taken for multiply "
//              << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
//                     .count()
//              << std::endl;
//    Tensor<densevec> res2 =
//        teov.multiply<densevec>(dlmop, left_c, left_b, right_c, right_b);
//    res2._infer_dimensionality();
//    t1 = std::chrono::high_resolution_clock::now();
//    res2._fill_values_only(teov, dlmop, left_c, left_b, right_c, right_b);
//    t2 = std::chrono::high_resolution_clock::now();
//    std::cout << "Time taken for fill_values_only densevec "
//              << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1)
//                     .count()
//              << std::endl;
//    CoOrdinate result_idx({});
//    res._infer_dimensionality();
//    result_idx.all_positions(res.get_dimensionality());
//    assert(IndexedTensor<densevec>(res, result_idx) ==
//           IndexedTensor<densevec>(res2, result_idx));
//    assert(IndexedTensor<densevec>(res2, result_idx) ==
//           IndexedTensor<densevec>(res, result_idx));
//}

int main() {
    //fill_values_only();
    //std::cout<<"Passed fill_values_only"<<std::endl;
    //fill_values_only_nobatch_scalar();
    //std::cout<<"Passed fill_values_only_nobatch_scalar"<<std::endl;
    symtensor_inf_shape();
    std::cout << "Passed symtensor_inf_shape" << std::endl;
    dense_opcount();
    std::cout << "Passed dense opcount and shape" << std::endl;
  dense_gemm_count();
  std::cout << "Passed dense_gemm_opcount" << std::endl;
  dense_gemm_shape();
  std::cout << "Passed dense_gemm_shape" << std::endl;
  sparse_gemm_count();
  std::cout << "Passed sparse_gemm_count" << std::endl;
  sparse_gemm_shape();
  std::cout << "Passed sparse_gemm_shape" << std::endl;
  fourd_tensor_contraction_count();
  std::cout << "Passed fourd_tensor_contraction_count" << std::endl;
  fourd_tensor_contraction_shape();
  std::cout << "Passed fourd_tensor_contraction_shape" << std::endl;
  dense_multiply();
  std::cout << "Passed dense_multiply" << std::endl;
  sparse_multiply();
  std::cout << "Passed sparse_multiply" << std::endl;
  sparse_multiply_offsetcrd();
  std::cout << "Passed sparse_multiply_offsetcrd" << std::endl;
  sparse_multiply_extrnonly();
  std::cout << "Passed sparse_multiply_extrnonly" << std::endl;
  teov_dlmop_opcount();
  std::cout << "Passed opcount teov * dlmop" << std::endl;
  tevv_dlmop_outputshape();
  std::cout << "Passed outputshape tevv * dlmop" << std::endl;

  return 0;
}
