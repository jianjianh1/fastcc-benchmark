#include "contract.hpp"
#include <chrono>
#include <iostream>
#include <iterator>

template <typename T> class FlopCounter {
private:
  int mults = 0;
  int adds = 0;

public:
  T mul(T a, T b) {
    mults++;
    return a * b;
  }
  T add(T a, T b) {
    adds++;
    return a + b;
  }
  int get_mults() { return mults; }
};

void dense_gemm_count() {
  // C(2, 2) = A(2, 5) * B(5, 2);
  int a_shape[2] = {2, 5};
  int b_shape[2] = {5, 2};
  Tensor A(10);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 5; j++) {
      int coords[2] = {i, j};
      A.get_nonzeros().push_back(NNZ(1.0, 2, coords));
    }
  }
  Tensor B(10);
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 2; j++) {
      int coords[2] = {i, j};
      B.get_nonzeros().push_back(NNZ(1.0, 2, coords));
    }
  }

  int left_contr[1] = {1};
  int right_contr[1] = {0};
  int num_mults = A.count_ops(B, 1, left_contr, right_contr);
  FlopCounter<float> counter;
  for (int i = 0; i < 2; i++) {
    for (int k = 0; k < 2; k++) {
      int accum = 0;
      for (int j = 0; j < 5; j++) {
        int left[2] = {i, j};
        int right[2] = {j, k};
        accum += counter.mul(A.get_valat(CoOrdinate(2, left)),
                             B.get_valat(CoOrdinate(2, right)));
      }
    }
  }
  assert(counter.get_mults() == num_mults);
  std::cout << "Count using FlopCounter " << counter.get_mults() << std::endl;
  std::cout << "Count using HT " << num_mults << std::endl;
}

void sparse_gemm_count() {
  // C(2, 2) = A(2, 5) * B(5, 2);
  int a_shape[2] = {2, 5};
  int b_shape[2] = {5, 2};
  Tensor A(10);
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 5; j++) {
      if (j % 2 != i)
        continue;
      int coords[2] = {i, j};
      A.get_nonzeros().push_back(NNZ(1.0, 2, coords));
    }
  }
  Tensor B(10);
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 2; j++) {
      if (j % 2 == i)
        continue;
      int coords[2] = {i, j};
      B.get_nonzeros().push_back(NNZ(1.0, 2, coords));
    }
  }
  int left_contr[1] = {1};
  int right_contr[1] = {0};
  int num_mults = A.count_ops(B, 1, left_contr, right_contr);
  FlopCounter<float> counter;
  for (int i = 0; i < 2; i++) {
    for (int k = 0; k < 2; k++) {
      int accum = 0;
      for (int j = 0; j < 5; j++) {
        int left[2] = {i, j};
        int right[2] = {j, k};
        float left_val = A.get_valat(CoOrdinate(2, left));
        float right_val = B.get_valat(CoOrdinate(2, right));
        if (left_val != -1 && right_val != -1)
          accum += counter.mul(left_val, right_val);
        // accum += counter.mul(A.get_valat(CoOrdinate(2, left)),
        //                      B.get_valat(CoOrdinate(2, right)));
      }
    }
  }
  assert(counter.get_mults() == num_mults);
  std::cout << "Count using FlopCounter " << counter.get_mults() << std::endl;
  std::cout << "Count using HT " << num_mults << std::endl;
}

void fourd_tensor_contraction() {
  // I(7, 11, 9, 11) = T0(7, 11, 10, 12, 9) * T1(7, 11, 10, 12, 11);
  int a_shape[5] = {7, 11, 10, 12, 9};
  int b_shape[5] = {7, 11, 10, 12, 11};
  Tensor A(100);
  int a_ctr = 0;
  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 11; j++) {
      for (int k = 0; k < 10; k++) {
        for (int l = 0; l < 12; l++) {
          for (int m = 0; m < 9; m++) {
            if (a_ctr == 100)
              break;

            int coords[5] = {i, j, k, l, m};
            A.get_nonzeros().push_back(NNZ(1.0, 5, coords));
            a_ctr++;
          }
        }
      }
    }
  }
  Tensor B(100);
  int b_ctr = 0;
  for (int i = 0; i < 7; i++) {
    for (int j = 0; j < 11; j++) {
      for (int k = 0; k < 10; k++) {
        for (int l = 0; l < 12; l++) {
          for (int m = 0; m < 11; m++) {
            if (b_ctr == 100)
              break;

            int coords[5] = {i, j, k, l, m};
            B.get_nonzeros().push_back(NNZ(1.0, 5, coords));
            b_ctr++;
          }
        }
      }
    }
  }
  int left_contr[2] = {2, 3};
  int right_contr[2] = {2, 3};
  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();
  int num_mults = A.count_ops(B, 2, left_contr, right_contr);
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  std::cout << "Time taken for count_ops " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << std::endl;
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
              float left_val = A.get_valat(CoOrdinate(5, left));
              float right_val = B.get_valat(CoOrdinate(5, right));
              if (left_val != -1 && right_val != -1)
                accum += counter.mul(left_val, right_val);
            }
          }
        }
      }
    }
  }
  std::cout << "Count using FlopCounter " << counter.get_mults() << std::endl;
  std::cout << "Count using HT " << num_mults << std::endl;
}

int main() {
  dense_gemm_count();
  sparse_gemm_count();
  fourd_tensor_contraction();
}
