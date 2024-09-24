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


void dlpno_bottleneck(Tensor<double> tevv, Tensor<double> t2_d_lmop) {
    //"TEvv_d_LMOP_T2_d_LMOP [b_mu, K, i, j, e_mu] = TEvv [b_mu, f_mu, K] * d_LMOP_T2_d_LMOP [i, j, e_mu, f_mu]";
    // contraction f_mu
    // batch, none
  auto left_c = CoOrdinate({1});
  auto right_c = CoOrdinate({3});
  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();
  auto res = tevv.multiply<double>(t2_d_lmop, left_c, CoOrdinate({}), right_c,
                                  CoOrdinate({}));
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  auto count_ops = tevv.count_ops(t2_d_lmop, left_c, right_c);
  auto time_taken =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  std::cout << "Time taken for mult " << time_taken <<"microsec"<< std::endl;
  std::cout << "Num ops " << count_ops << std::endl;
  std::cout << "Intensity " << double(count_ops) / time_taken<<"MFLOP/s" << std::endl;
}

void dlpno_bottleneck2(Tensor<double> tevv, Tensor<densevec> d_lmop) {
    //"TEvv_d_LMOP [b_mu, K, i, j, f_ij] = TEvv [b_mu, f_mu, K] * d_LMOP [i, j, f_mu, f_ij]";
    // contraction f_mu
    // batch, none
  auto left_c = CoOrdinate({1});
  auto right_c = CoOrdinate({2});
  std::chrono::high_resolution_clock::time_point t1 =
      std::chrono::high_resolution_clock::now();
  auto res = tevv.multiply<densevec>(d_lmop, left_c, CoOrdinate({}), right_c,
                                  CoOrdinate({}));
  std::chrono::high_resolution_clock::time_point t2 =
      std::chrono::high_resolution_clock::now();
  auto time_taken =
      std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
  std::cout << "Time taken for mult " << time_taken <<"microsec"<< std::endl;
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

  Tensor<densemat> res_vanilla =
      t2.multiply<densemat, densemat>(t2, CoOrdinate({}), CoOrdinate({0, 1}),
                                      CoOrdinate({}), CoOrdinate({0, 1}));
  res_vanilla._infer_dimensionality();
  res_vanilla._infer_shape();
  res_vanilla.write("T2_out_vanilla.tns");

  std::cout << "Wrote the ground truth" << std::endl;

  Tensor<densemat> res(t2.get_size());
  tq.addContraction(res, t2, t2, CoOrdinate({}), CoOrdinate({0, 1}),
                    CoOrdinate({}), CoOrdinate({0, 1}));
  tq.run();
  res._infer_dimensionality();
  res._infer_shape();
  res.write("T2_out.tns");
  // diff the two files on disk, should be equal
}

void teov_dlmop_opcount(Tensor<double> dteov, Tensor<densevec> dlmop){
    // [j, K, i, b_ij] = (TEov[[j, b_mu, K]] * d_LMOP[[i, j, b_mu, b_ij]]) @ 88242
    auto left_c = CoOrdinate(std::vector<int>({0, 1})); //batch,contraction
    //auto left_c = CoOrdinate(std::vector<int>({1, 0})); //contraction,batch
    auto right_c = CoOrdinate(std::vector<int>({1, 2})); //batch,contraction
    //auto right_c = CoOrdinate(std::vector<int>({2, 1})); //contraction,batch
    auto count_ops = dteov.count_ops(dlmop, left_c, right_c);
    std::cout << "Num ops " << count_ops << std::endl;
}

void teov_dlmop_multiply(Tensor<double> dteov, Tensor<densevec> dlmop){
    // [j, K, i, b_ij] = (TEov[[j, b_mu, K]] * d_LMOP[[i, j, b_mu, b_ij]]) @ 88242
    auto left_c = CoOrdinate(std::vector<int>({1}));
    auto left_b = CoOrdinate(std::vector<int>({0}));
    auto right_c = CoOrdinate(std::vector<int>({2}));
    auto right_b = CoOrdinate(std::vector<int>({0}));
    Tensor<densevec> teov_dlmop = dteov.multiply<densevec>(dlmop, left_c, left_b, right_c, right_b);
    teov_dlmop._infer_dimensionality();
    teov_dlmop._infer_shape();
    teov_dlmop.write("TEov_d_LMOP.tns");
    TaskQueue tq;
    Tensor<densevec> teov_dlmop_tq;
    tq.addContraction(teov_dlmop_tq, dteov, dlmop, left_c, left_b,
                      right_c, right_b);
    tq.run();
    teov_dlmop_tq._infer_dimensionality();
    teov_dlmop_tq._infer_shape();
    teov_dlmop_tq.write("TEov_d_LMOP_tq.tns");
}

SymbolicTensor tevv_dlmop_outputshape(Tensor<double> dtevv, Tensor<densevec> dlmop) {
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

Tensor<densevec> tevv_dlmop_contraction(Tensor<double> dtevv, Tensor<densevec> dlmop) {
    // [b_mu, K, m, i, e_mi] = (TEvv[[b_mu, e_mu, K]] * d_LMOP[[m, i, e_mu,
    // e_mi]]) contract e_mu
    auto left_c = CoOrdinate(std::vector<int>({1}));
    auto left_b = CoOrdinate(std::vector<int>({}));
    auto right_c = CoOrdinate(std::vector<int>({2}));
    auto right_b = CoOrdinate(std::vector<int>({}));
    auto start = std::chrono::high_resolution_clock::now();
    Tensor<densevec> output_tensor = dtevv.multiply<densevec>(
        dlmop, left_c, left_b, right_c, right_b);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken for mult "
              << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                       start)
                     .count()
              << " microseconds" << std::endl;
    return output_tensor;
}

Tensor<densevec> tevv_dlmop_fillvalues(Tensor<double> dtevv, Tensor<densevec> dlmop) {
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
    return result;
}

void tevv_dlmop_taskq() {
    TaskQueue tq;
    Tensor<double> dtevv("TEvv.tns", true);
    Tensor<densevec> dlmop("d_LMOP.tns", true);
    Tensor<densevec> dlmop_out(dlmop.get_size());
    tq.addContraction(dlmop_out, dtevv, dlmop, CoOrdinate({1}), CoOrdinate({}),
                      CoOrdinate({2}), CoOrdinate({}));
    auto start = std::chrono::high_resolution_clock::now();
    tq.loopUntil();
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken for mult in tq"
              << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                       start)
                     .count()
              << " microseconds" << std::endl;
}

void task_queue_loop() {
    // check for mem leak
    TaskQueue tq;
    Tensor<densemat> t2("T2.tns", true);
    Tensor<densemat> res_vanilla =
        t2.multiply<densemat, densemat>(t2, CoOrdinate({}), CoOrdinate({0, 1}),
                                        CoOrdinate({}), CoOrdinate({0, 1}));
    res_vanilla._infer_dimensionality();
    res_vanilla._infer_shape();
    res_vanilla.write("T2_out_vanilla.tns");
    std::cout << "wrote ground truth" << std::endl;
    Tensor<densemat> res(t2.get_size());
    tq.addContraction(res, t2, t2, CoOrdinate({}), CoOrdinate({0, 1}),
                      CoOrdinate({}), CoOrdinate({0, 1}));
    tq.updateDoubles(&res);
    tq.loopUntil();
    t2.delete_old_values();
    res.delete_old_values();

    Tensor<densemat> tq_result = tq.getDoubles();
    tq_result._infer_dimensionality();
    tq_result._infer_shape();
    tq_result.write("T2_out.tns");


    // diff the two files on disk, should be equal
}

void dlpno_bottleneck3(Tensor<double> tevv, Tensor<densemat> TEov_d_LMOP_T1_S_iikl){
    //Contracting TEvv[[a_mu, e_mu, K]] and TEov_d_LMOP_T1_S_iikl[[i, j, K, f_ij, b_ij]] at [2] and [2], to form TEvv_TEov_d_LMOP_T1_S_iikl[[a_mu, e_mu, i, j, f_ij, b_ij]]. Batch indices are at [] and []
    //
    auto left_c = CoOrdinate(std::vector<int>({2}));
    auto right_c = CoOrdinate(std::vector<int>({2}));
    std::chrono::high_resolution_clock::time_point t1 =
        std::chrono::high_resolution_clock::now();
    auto res = tevv.multiply<densemat>(TEov_d_LMOP_T1_S_iikl, left_c, CoOrdinate({}), right_c,
                                    CoOrdinate({}));
    std::chrono::high_resolution_clock::time_point t2 =
        std::chrono::high_resolution_clock::now();
    std::cout<<"Time taken for mult "<<std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()<<"microsec"<<std::endl;
    return;
}

int main() {
    Tensor<double> teov("TEov.tns", true);
    Tensor<densevec> dlmop("d_LMOP.tns", true);
    teov_dlmop_multiply(teov, dlmop);
     //task_queue_loop();
    //Tensor<double> tevv("TEvv.tns", true);
    //Tensor<densemat> teov_something;
    //read_from_dump(teov_something, "debug_TEov_d_LMOP_T1_S_iikl.tns");
    //dlpno_bottleneck3(tevv, teov_something);
}
