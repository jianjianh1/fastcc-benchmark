#include "contract.hpp"
#include "read.hpp"
#include "task_queue.hpp"
#include <chrono>
#include <iostream>
#include <iterator>


void fourd_tensor_contraction_numeric() {
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
  auto left_c = CoOrdinate({2, 3});
  auto left_b = CoOrdinate({0, 1});
  auto right_c = CoOrdinate({2, 3});
  auto right_b = CoOrdinate({0, 1});
  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();
  auto res_tensor = A.multiply<float>(B, left_c, left_b, right_c, right_b);
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  std::cout
      << "Time taken for mult "
      << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
      << std::endl;
  //FlopCounter<float> counter;
  //for (int i = 0; i < 7; i++) {
  //  for (int j = 0; j < 11; j++) {
  //    for (int k = 0; k < 9; k++) {
  //      for (int l = 0; l < 11; l++) {
  //        int accum = 0;
  //        for (int m = 0; m < 10; m++) {
  //          for (int n = 0; n < 12; n++) {
  //            int left[5] = {i, j, m, n, k};
  //            int right[5] = {i, j, m, n, l};
  //            auto leftcord = CoOrdinate(5, left);
  //            auto rightcord = CoOrdinate(5, right);
  //            float left_val = A.get_valat(leftcord);
  //            float right_val = B.get_valat(rightcord);
  //            if (left_val != -1 && right_val != -1)
  //              accum += counter.mul(left_val, right_val);
  //          }
  //        }
  //      }
  //    }
  //  }
  //}
  //std::cout << "Count using FlopCounter " << counter.get_mults() << std::endl;
  //std::cout << "Count using HT " << num_mults << std::endl;
}

void dlpno_4cint(Tensor<float> teov) {
  auto left_c = CoOrdinate({2});
  auto right_c = CoOrdinate({2});
  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();
  auto res = teov.multiply<float>(teov, left_c, CoOrdinate({}), right_c,
                                  CoOrdinate({}));
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  auto count_ops = teov.count_ops(teov, left_c, right_c);
  auto time_taken =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  std::cout << "Time taken for mult " << time_taken <<"microsec"<< std::endl;
  std::cout << "Num ops " << count_ops << std::endl;
  std::cout << "Intensity " << double(count_ops) / time_taken<<"MFLOP/s" << std::endl;
}



void sparse_gemm(Tensor<float> some) {
  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();
  // auto output_coordinates = some.output_shape(
  //     some, CoOrdinate({1}), CoOrdinate({}), CoOrdinate({0}),
  //     CoOrdinate({}));
  // auto num_ops = some.count_ops(some, CoOrdinate({1}), CoOrdinate({0}));
  auto output_matrix = some.multiply<float>(
      some, CoOrdinate({1}), CoOrdinate({}), CoOrdinate({0}), CoOrdinate({}));
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  std::cout
      << "Time taken for get_contraction_shape "
      << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
      << " microseconds" << std::endl;
  // std::cout << " Gonna write " << output_matrix.size()
  //           << " coordinates to file" << std::endl;
  std::string filename = "output_coordinates.txt";
  std::ofstream file(filename);
  for (auto &nnz : output_matrix.get_nonzeros()) {
    file << nnz.to_string() << std::endl;
  }
}

void t2_pno_dgemm() {
  Tensor<densemat> t2("T2.tns", true);
  Tensor<densemat> res_lmo_lmo =
      t2.multiply<densemat, densemat>(t2, CoOrdinate({}), CoOrdinate({0, 1}),
                                      CoOrdinate({}), CoOrdinate({0, 1}));
  res_lmo_lmo.write("T2_out.tns");
}

void t1_pno_vecinner() {
  Tensor<densevec> t1("T1.tns", true);
  Tensor<double> res_lmo_lmo = t1.multiply<double, densevec>(
      t1, CoOrdinate({}), CoOrdinate({0}), CoOrdinate({}), CoOrdinate({0}));
  std::cout << "Num ops " << t1.count_ops(t1, CoOrdinate({0}), CoOrdinate({0}))
            << std::endl;
  res_lmo_lmo.write("T1_out.tns");
}

void task_queue() {
  TaskQueue tq;
  Tensor<densemat> t2("T2.tns", true);
  Tensor<densemat> res(t2.get_size());
  tq.addContraction(res, t2, t2, CoOrdinate({}), CoOrdinate({0, 1}), CoOrdinate({}),
                    CoOrdinate({0, 1}));
  tq.run();
  res.write("T2_out.tns");
  Tensor<densemat> res_vanilla = t2.multiply<densemat, densemat>(
      t2, CoOrdinate({}), CoOrdinate({0, 1}), CoOrdinate({}), CoOrdinate({0, 1}));
  res_vanilla.write("T2_out_vanilla.tns");
}

void task_queue_loop() {
    //check for mem leak
  TaskQueue tq;
  Tensor<densemat> t2("T2.tns", true);
  Tensor<densemat> res(t2.get_size());
  tq.addContraction(res, t2, t2, CoOrdinate({}), CoOrdinate({0, 1}), CoOrdinate({}),
                    CoOrdinate({0, 1}));
  tq.updateDoubles(res);
  tq.loopUntil();
  t2.delete_old_values();
  res.delete_old_values();
  //res.write("T2_out.tns");
  //Tensor<densemat> res_vanilla = t2.multiply<densemat, densemat>(
  //    t2, CoOrdinate({}), CoOrdinate({0, 1}), CoOrdinate({}), CoOrdinate({0, 1}));
  //res_vanilla.write("T2_out_vanilla.tns");
}

int main() {
//task_queue_loop();
    Tensor<float> teov("TEov.tns", true);
    dlpno_4cint(teov);
}
