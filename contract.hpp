#ifndef CONTRACT_HPP
#define CONTRACT_HPP
#include "absl/container/flat_hash_map.h"
#include "coordinate.hpp"
#include "taskflow.hpp"
#include "timer.hpp"
#include "index_tensors.hpp"
#include <algorithm>
#include <ankerl/unordered_dense.h>
#include <omp.h>
#include <atomic>
#include <boost/functional/hash.hpp>
#include <chrono>
#include <emhash/hash_table8.hpp>
#include <forward_list>
#include <iostream>
#include <ranges>
#include <tsl/hopscotch_map.h>
#include <tsl/hopscotch_set.h>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

template <class DT> class Tensor {
private:
  std::vector<NNZ<DT>> nonzeros;
  int *shape;
  int dimensionality = 42;
  using hashmap_counts = tsl::hopscotch_map<CoOrdinate, int>;
  using hashmap_shape =
      tsl::hopscotch_map<CoOrdinate,
                         std::vector<std::pair<CoOrdinate, CoOrdinate>>>;
  using hashmap_vals = tsl::hopscotch_map<CoOrdinate, DT>;

public:
  using iterator = typename std::vector<NNZ<DT>>::iterator;
  using value_type = typename std::vector<NNZ<DT>>::value_type;
  iterator begin() { return nonzeros.begin(); }
  iterator end() { return nonzeros.end(); }
  Tensor(std::string fname, bool);
  double reduce() {
    double sum = 0;
    for (auto &nnz : nonzeros) {
      sum += nnz.get_data();
    }
    return sum;
  }
  void write(std::string fname);
  int get_dimensionality() {
    if (dimensionality == 42) {
      this->_infer_dimensionality();
    }
    return dimensionality;
  }
  // Constructor for a tensor of given shape and number of non-zeros, fills
  // with random values and indices
  Tensor(int size, int dimensionality, int *shape) {
    this->shape = shape;
    this->dimensionality = dimensionality;
    nonzeros.reserve(size);
    for (int i = 0; i < size; i++) {
      nonzeros.emplace_back(dimensionality, shape);
    }
  }
  void delete_old_values() {
    // TODO: might be a leak....but need to replace with smart pointers maybe.
    // if constexpr (std::is_class<DT>::value) {
    //   for (auto &nnz : nonzeros) {
    //     nnz.get_data().free();
    //   }
    // }
    nonzeros.clear();
  }
  // Make a tensor with just ones at given positions
  template <class It> Tensor(It begin, It end) {
    for (auto it = begin; it != end; it++) {
      if constexpr (std::is_class<DT>::value) {
        nonzeros.emplace_back(DT(), *it);
      } else {
        nonzeros.emplace_back(1.0, *it);
      }
    }
    this->_infer_dimensionality();
    this->_infer_shape();
  }
  Tensor(SymbolicTensor &sym) {
    for (auto &cord : sym) {
      if constexpr (std::is_class<DT>::value) {
        nonzeros.emplace_back(DT(), cord);
      } else {
        nonzeros.emplace_back(0.0, cord);
      }
    }
    this->_infer_dimensionality();
    this->_infer_shape();
  }
  Tensor(int size = 0) { nonzeros.reserve(size); }
  std::vector<NNZ<DT>> &get_nonzeros() { return nonzeros; }
  int get_size() { return nonzeros.size(); }
  void _infer_dimensionality() {
    if (nonzeros.size() > 0) {
      dimensionality = nonzeros[0].get_coords().get_dimensionality();
    }
  }
  std::string get_shape_string() {
    if (dimensionality == 42) {
      this->_infer_dimensionality();
      this->_infer_shape();
    }
    std::string str = "";
    for (int i = 0; i < dimensionality; i++) {
      str += std::to_string(shape[i]) + " ";
    }
    return str;
  }
  int *get_shape_ref() {
    if (dimensionality == 42) {
      this->_infer_dimensionality();
      this->_infer_shape();
    }
    return shape;
  }
  void _infer_shape() {
    // TODO this is a mem-leak. Add a guard before allocation
    if (nonzeros.size() > 0) {
      shape = new int[dimensionality];
      for (int i = 0; i < dimensionality; i++) {
        shape[i] = 0;
      }
      for (auto &nnz : nonzeros) {
        auto coords = nnz.get_coords();
        for (int i = 0; i < dimensionality; i++) {
          if (coords.get_index(i) > shape[i]) {
            shape[i] = coords.get_index(i);
          }
        }
      }
      std::vector<int> shape_vec(shape, shape + dimensionality);
      for (auto &nnz : nonzeros) {
        nnz.get_coords().set_shape(shape_vec);
      }
    }
  }
  DT get_valat(CoOrdinate coords) {
    // TODO: merge this with the operator[]
    for (auto &nnz : nonzeros) {
      auto this_coords = nnz.get_coords();
      if (this_coords == coords) {
        return nnz.get_data();
      } else {
      }
    }
    std::cerr << "Error, trying to access a non-existent coordinate"
              << std::endl;
    exit(1);
  }

  // Returns a set of coordinates that are the result of the contraction of
  // two tensors. order is: (batch indices, left external indices, right
  // external indices). order within batch indices is dependent on the left
  // operand
  tsl::hopscotch_set<CoOrdinate>
  output_shape(Tensor &other, CoOrdinate left_contr, CoOrdinate left_batch,
               CoOrdinate right_contr, CoOrdinate right_batch) {
    auto right_symbolic = SymbolicTensor(other);
    return SymbolicTensor(*this).output_shape(
        right_symbolic, left_contr, left_batch, right_contr, right_batch);
  }

  // does not multiply data, just returns the coordinates
  Tensor contract(Tensor &other, CoOrdinate left_contraction,
                  CoOrdinate left_batch, CoOrdinate right_contraction,
                  CoOrdinate right_batch) {
    tsl::hopscotch_set<CoOrdinate> output_coords = this->output_shape(
        other, left_contraction, left_batch, right_contraction, right_batch);
    return Tensor(output_coords.begin(), output_coords.end());
  }

  template <class Right>
  std::pair<uint64_t, uint64_t> bound_output_nnz_inner_outer(Tensor<Right> &other,
                                                 CoOrdinate left_contr,
                                                 CoOrdinate right_contr) {
    // first compute number of FLOPs
    std::vector<int> all_indices = std::vector<int>(this->get_dimensionality());
    std::iota(all_indices.begin(), all_indices.end(), 0);
    CoOrdinate left_external = CoOrdinate(all_indices).remove(left_contr);
    SmallIndexedTensor<DT> left_indexed =
        SmallIndexedTensor<DT>(*this, left_external);
    SmallIndexedTensor<Right> right_indexed =
        SmallIndexedTensor<Right>(other, right_contr);
    uint64_t flop_count = 0;
    uint64_t lower_bound = 0;
    uint64_t upper_bound = 0;
    for (auto &left_slice : left_indexed.indexed_tensor) {
      uint64_t row_nnz_lb = 0;
      uint64_t row_nnz_ub = 0;
      for (auto &left_pair : left_slice.second) {
        flop_count += right_indexed.row_size_of(left_pair.first);
        row_nnz_lb += right_indexed.row_size_of(left_pair.first);
        row_nnz_ub += right_indexed.row_size_of(left_pair.first);
      }
      row_nnz_lb = row_nnz_lb / left_slice.second.size() +
                         (row_nnz_lb % left_slice.second.size() != 0);
      row_nnz_ub = std::min(row_nnz_ub, right_indexed.get_linearization_bound());
      upper_bound += row_nnz_ub;
      lower_bound += row_nnz_lb;
    }
    // lower bound is F/NNZ_l
    // upper bound is F
    return {lower_bound, upper_bound};
  }

  float avg_nnz_per_tile(CoOrdinate contr_pos, int tile_size){
    TileIndexedTensor<DT> left_indexed =
        TileIndexedTensor<DT>(*this, contr_pos, tile_size);
    float avg = 0.0;
    for(int i = 0; i < left_indexed.num_tiles(); i++){
      avg += left_indexed.num_nnz_in_tile(i);
    }
    avg /= left_indexed.num_tiles();
    return avg;
  }

  float avg_nnz_per_active_column(CoOrdinate contr_pos, int tile_size){
    TileIndexedTensor<DT> left_indexed =
        TileIndexedTensor<DT>(*this, contr_pos, tile_size);
    float avg = 0.0;
    int iter = 0;
    for(int i = 0; i < left_indexed.num_tiles(); i++){
        auto this_cnt = left_indexed.nnz_per_active_column(i);
      avg += this_cnt;
      if(this_cnt > 0) iter++;
    }
    avg /= iter;
    return avg;
  }
  float avg_active_columns(CoOrdinate contr_pos, int tile_size){
    TileIndexedTensor<DT> left_indexed =
        TileIndexedTensor<DT>(*this, contr_pos, tile_size);
    float avg = 0.0;
    int iter = 0;
    for(int i = 0; i < left_indexed.num_tiles(); i++){
        auto this_cnt = left_indexed.num_active_columns(i);
        if(this_cnt > 0) iter++;
      avg += this_cnt;
    }
    avg /= iter;
    return avg;
  }

  uint64_t total_active_columns(CoOrdinate contr_pos, int tile_size){
    TileIndexedTensor<DT> left_indexed =
        TileIndexedTensor<DT>(*this, contr_pos, tile_size);
    uint64_t total = 0;
    for(int i = 0; i < left_indexed.num_tiles(); i++){
        total += left_indexed.num_active_columns(i);
    }
    return total;
  }

  template <class Right>
  std::pair<uint64_t, uint64_t> bound_output_nnz_outer_outer(Tensor<Right> &other,
                                                 CoOrdinate left_contr,
                                                 CoOrdinate right_contr) {
    // first compute number of FLOPs
    SmallIndexedTensor<DT> left_indexed =
        SmallIndexedTensor<DT>(*this, left_contr);
    SmallIndexedTensor<Right> right_indexed =
        SmallIndexedTensor<Right>(other, right_contr);
    uint64_t lower_bound = 0;
    uint64_t upper_bound = 0;
    for (auto &left_slice : left_indexed.indexed_tensor) {
      auto &right_slice = right_indexed.indexed_tensor[left_slice.first];
      uint64_t this_size = left_slice.second.size() * right_slice.size();
      upper_bound += this_size;
      if (lower_bound < this_size)
        lower_bound = this_size;
    }
    return {lower_bound, upper_bound};
  }

  // Needs shape for left and right tensors
  // full outer multiplication
  template <class RES, class RIGHT>
  Tensor<RES> multiply(Tensor<RIGHT> &other, CoOrdinate left_contr,
                       CoOrdinate left_batch, CoOrdinate right_contr,
                       CoOrdinate right_batch) {
    std::chrono::high_resolution_clock::time_point start, end;
    start = std::chrono::high_resolution_clock::now();
    CoOrdinate left_idx_pos = CoOrdinate(left_batch, left_contr);
    IndexedTensor<DT> left_indexed = IndexedTensor<DT>(*this, left_idx_pos);
    CoOrdinate right_idx_pos = CoOrdinate(right_batch, right_contr);
    IndexedTensor<RIGHT> right_indexed =
        IndexedTensor<RIGHT>(other, right_idx_pos);

    // tsl::hopscotch_map<OutputCoordinate, RES, std::hash<OutputCoordinate>,
    // std::equal_to<OutputCoordinate>,
    // std::allocator<std::pair<OutputCoordinate, RES>>, (unsigned int)62,
    // (bool)0, tsl::hh::prime_growth_policy> result;
    tsl::hopscotch_map<OutputCoordinate, RES> result;

    // tsl::hopscotch_map<OutputCoordinate, RES, ...,
    // GrowthPolicy=tsl::hh::prime_growth_policy> result;
    std::vector<int> batch_pos_afterhash(left_batch.get_dimensionality());
    std::iota(batch_pos_afterhash.begin(), batch_pos_afterhash.end(), 0);
    BoundedPosition batchpos = BoundedPosition(batch_pos_afterhash);
    end = std::chrono::high_resolution_clock::now();
    double time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Time taken to index: " << time_taken << std::endl;
    long long int counter = 0;

    for (auto &left_entry : left_indexed.indexed_tensor) {
      auto right_entry = right_indexed.indexed_tensor.find(left_entry.first);
      if (right_entry != right_indexed.indexed_tensor.end()) {

        for (auto &left_ev :
             left_entry.second) { // loop over (e_l, nnz_l): external
                                  // left, nnz at that external left.
          for (auto &right_ev : right_entry->second) {
            BoundedCoordinate left_bc = left_entry.first;

            BoundedCoordinate batch_coords = left_bc.gather(
                batchpos); // assumes that batch positions are leftmost,
                           // so they will work with a left subset.
            BoundedCoordinate left_external = left_ev.first;
            BoundedCoordinate right_external = right_ev.first;

            // CoOrdinate output_coords =
            //     CoOrdinate(batch_coords, left_ev.first,
            //     right_ev.first);
            OutputCoordinate output_coords =
                OutputCoordinate(batch_coords, left_external, right_external);
            RES outp;
            outp = left_ev.second * right_ev.second;
            auto result_ref = result.find(output_coords);
            counter++;
            if (result_ref != result.end()) {
              result_ref.value() += outp;
            } else {
              result[output_coords] = outp;
            }
          }
        }
      }
    }
    std::cout << "Overflow size " << result.overflow_size() << std::endl;
    std::cout << "Result NNZ count " << result.size() << std::endl;
    std::cout << "Number of calls to find " << counter << std::endl;
    std::cout << "Number of calls to == "
              << OutputCoordinate::get_equality_count() << std::endl;
    start = std::chrono::high_resolution_clock::now();
    Tensor<RES> result_tensor(result.size());

    for (auto nnz : result) {
      result_tensor.get_nonzeros().push_back(
          NNZ<RES>(nnz.second, nnz.first.merge()));
    }
    end = std::chrono::high_resolution_clock::now();
    time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Time taken to writeback: " << time_taken << std::endl;
    return result_tensor;
  }

  // Needs shape for left and right tensors
  // full outer multiplication
  template <class RES, class RIGHT>
  CompactTensor<RES> outer_outer_multiply(Tensor<RIGHT> &other,
                                          CoOrdinate left_contr,
                                          CoOrdinate right_contr) {
    int result_dimensionality =
        this->get_dimensionality() + other.get_dimensionality() -
        (left_contr.get_dimensionality() + right_contr.get_dimensionality());
    std::cout << "Result dimensionality: " << result_dimensionality
              << std::endl;
    std::chrono::high_resolution_clock::time_point start, end;
    start = std::chrono::high_resolution_clock::now();
    SmallIndexedTensor<DT> left_indexed =
        SmallIndexedTensor<DT>(*this, left_contr);
    uint64_t left_inner_max = left_indexed.get_linearization_bound();
    SmallIndexedTensor<RIGHT> right_indexed =
        SmallIndexedTensor<RIGHT>(other, right_contr);
    uint64_t right_inner_max = right_indexed.get_linearization_bound();

    DT *accumulator =
        (DT *)malloc((left_inner_max + 1) * (right_inner_max + 1) * sizeof(DT));
    std::fill(accumulator,
              accumulator + (left_inner_max + 1) * (right_inner_max + 1), DT());
    if (accumulator == nullptr) {
      std::cerr << "Failed to allocate memory for accumulator" << std::endl;
      exit(1);
    } else {
      std::cout << "Allocated " << (left_inner_max + 1) * (right_inner_max + 1)
                << " elts for accumulator" << std::endl;
    }
    end = std::chrono::high_resolution_clock::now();
    double time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Time taken to index: " << time_taken << std::endl;
    start = std::chrono::high_resolution_clock::now();

    for (auto &left_entry : left_indexed.indexed_tensor) {
      auto right_entry = right_indexed.indexed_tensor.find(left_entry.first);
      if (right_entry != right_indexed.indexed_tensor.end()) {
        for (auto &left_ev :
             left_entry.second) { // loop over (e_l, nnz_l): external
                                  // left, nnz at that external left.
          for (auto &right_ev : right_entry->second) {
            accumulator[left_ev.first * (right_inner_max + 1) +
                        right_ev.first] += left_ev.second * right_ev.second;
          }
        }
      }
    }
    end = std::chrono::high_resolution_clock::now();
    time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Time taken to contract: " << time_taken << std::endl;
    BoundedCoordinate sample_left = this->nonzeros[0]
                                        .get_coords()
                                        .remove(left_contr)
                                        .get_bounded(this->get_shape_ref());
    BoundedCoordinate sample_right = other.nonzeros[0]
                                         .get_coords()
                                         .remove(right_contr)
                                         .get_bounded(other.get_shape_ref());
    start = std::chrono::high_resolution_clock::now();
    CompactTensor<RES> result_tensor = CompactTensor<RES>(
        (left_inner_max + 1) * (right_inner_max + 1), result_dimensionality);
    for (uint64_t i = 0; i < left_inner_max + 1; i++) {
      for (uint64_t j = 0; j < right_inner_max + 1; j++) {
        if (accumulator[i * (right_inner_max + 1) + j] == DT())
          continue;
        CompactCordinate this_cord =
            CompactCordinate(i, sample_left, j, sample_right, 0);
        result_tensor.push_nnz(accumulator[i * (right_inner_max + 1) + j],
                               this_cord);
      }
    }
    end = std::chrono::high_resolution_clock::now();
    time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Time taken to writeback: " << time_taken << std::endl;
    std::cout << "Got " << result_tensor.get_nnz_count() << " nonzeros"
              << std::endl;
    return result_tensor;
  }

template <class RES, class RIGHT>
  ListTensor<RES>
  parallel_tiled_outer_multiply(Tensor<RIGHT> &other, CoOrdinate left_contr,
                       CoOrdinate right_contr, int tile_size = 0) {
    // for l_T
    //    for c
    //        for r
    //            for T_l
    int result_dimensionality =
        this->get_dimensionality() + other.get_dimensionality() -
        (left_contr.get_dimensionality() + right_contr.get_dimensionality());
    BoundedCoordinate sample_left = this->nonzeros[0]
                                        .get_coords()
                                        .remove(left_contr)
                                        .get_bounded(this->get_shape_ref());
    BoundedCoordinate sample_right = other.nonzeros[0]
                                         .get_coords()
                                         .remove(right_contr)
                                         .get_bounded(other.get_shape_ref());
    std::cout << "Result dimensionality: " << result_dimensionality
              << std::endl;
    std::chrono::high_resolution_clock::time_point start, end;
    start = std::chrono::high_resolution_clock::now();
    // get LLC size in bytes
    //uint64_t llc_size = 16 * 1024 * 1024 / sizeof(DT);
    int num_workers = std::thread::hardware_concurrency()/ 2;
    tf::Taskflow taskflow;
    tf::Executor executor(num_workers);

    SmallIndexedTensor<RIGHT> right_indexed =
        SmallIndexedTensor<RIGHT>(other, right_contr);
    uint64_t right_inner_max = right_indexed.get_linearization_bound();
    //tile_size = llc_size / (right_inner_max + 1);
    TileIndexedTensor<DT> left_indexed =
        TileIndexedTensor<DT>(*this, left_contr);
    uint64_t left_inner_max = left_indexed.tile_size;

    DT *thread_local_accumulators[num_workers];
    ListTensor<RES> thread_local_results[num_workers];
    for (int i = 0; i < num_workers; i++) {
      thread_local_accumulators[i] =
          (DT *)malloc((left_inner_max) * (right_inner_max + 1) * sizeof(DT));
      if (thread_local_accumulators[i] == nullptr) {
        std::cerr << "Failed to allocate memory for accumulator" << std::endl;
        exit(1);
      } else {
        std::cout << "Allocated " << (left_inner_max) * (right_inner_max + 1)
                  << " elts for accumulator" << std::endl;
      }
    }
    end = std::chrono::high_resolution_clock::now();
    double time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Time taken to index: " << time_taken << std::endl;
    start = std::chrono::high_resolution_clock::now();

    for (auto &left_tile : left_indexed.indexed_tensor) {
      taskflow.emplace([&]() mutable {
        DT *myacc = thread_local_accumulators[executor.this_worker_id()];
        std::fill(myacc, myacc + left_inner_max * (right_inner_max + 1), DT());
        for (const auto &left_entry : left_tile.second) {
          auto right_entry =
              right_indexed.indexed_tensor.find(left_entry.first);
          if (right_entry != right_indexed.indexed_tensor.end()) {
            for (auto &left_ev :
                 left_entry.second) { // loop over (e_l, nnz_l): external
                                      // left, nnz at that external left.
              for (auto &right_ev : right_entry->second) {
                myacc[left_ev.first * (right_inner_max + 1) + right_ev.first] +=
                    left_ev.second * right_ev.second;
              }
            }
          }
        }
        // drain here.
        for (uint64_t i = 0; i < left_inner_max; i++) {
          for (uint64_t j = 0; j < right_inner_max + 1; j++) {
            if (myacc[i * (right_inner_max + 1) + j] == DT())
              continue;
            uint64_t left_index =
                left_indexed.get_linear_index(left_tile.first, i);
            CompactCordinate this_cord =
                CompactCordinate(left_index, sample_left, j, sample_right);
            thread_local_results[executor.this_worker_id()].push_nnz(
                myacc[i * (right_inner_max + 1) + j], this_cord);
          }
        }
      });
    }
    executor.run(taskflow).wait();
    end = std::chrono::high_resolution_clock::now();
    time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Time taken to contract: " << time_taken << std::endl;

    start = std::chrono::high_resolution_clock::now();
    ListTensor<RES> result_tensor = thread_local_results[0];
    int iter = 0;
    for(auto &local_res: thread_local_results){
        if(iter++ == 0) continue;
        result_tensor.concatenate(local_res);
    }
    end = std::chrono::high_resolution_clock::now();
    end = std::chrono::high_resolution_clock::now();
    time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Time taken to writeback: " << time_taken << std::endl;
    std::cout << "Got " << result_tensor.compute_nnz_count() << " nonzeros"
              << std::endl;
    return result_tensor;
  }

  template <class RES, class RIGHT>
  ListTensor<RES>
  tiled_outer_multiply(Tensor<RIGHT> &other, CoOrdinate left_contr,
                       CoOrdinate right_contr, int tile_size = 0) {
    // for l_T
    //    for c
    //        for r
    //            for T_l
    int result_dimensionality =
        this->get_dimensionality() + other.get_dimensionality() -
        (left_contr.get_dimensionality() + right_contr.get_dimensionality());
    BoundedCoordinate sample_left = this->nonzeros[0]
                                        .get_coords()
                                        .remove(left_contr)
                                        .get_bounded(this->get_shape_ref());
    BoundedCoordinate sample_right = other.nonzeros[0]
                                         .get_coords()
                                         .remove(right_contr)
                                         .get_bounded(other.get_shape_ref());
    std::cout << "Result dimensionality: " << result_dimensionality
              << std::endl;
    std::chrono::high_resolution_clock::time_point start, end;
    start = std::chrono::high_resolution_clock::now();
    // get LLC size in bytes
    //uint64_t llc_size = 16 * 1024 * 1024 / sizeof(DT);

    SmallIndexedTensor<RIGHT> right_indexed =
        SmallIndexedTensor<RIGHT>(other, right_contr);
    uint64_t right_inner_max = right_indexed.get_linearization_bound();
    //tile_size = llc_size / (right_inner_max + 1);
    TileIndexedTensor<DT> left_indexed =
        TileIndexedTensor<DT>(*this, left_contr);
    uint64_t left_inner_max = left_indexed.tile_size;

    DT *accumulator =
        (DT *)malloc((left_inner_max) * (right_inner_max + 1) * sizeof(DT));
    if (accumulator == nullptr) {
      std::cerr << "Failed to allocate memory for accumulator" << std::endl;
      exit(1);
    } else {
      std::cout << "Allocated " << (left_inner_max) * (right_inner_max + 1)
                << " elts for accumulator" << std::endl;
    }
    end = std::chrono::high_resolution_clock::now();
    double time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Time taken to index: " << time_taken << std::endl;
    ListTensor<RES> result_tensor(result_dimensionality);
    start = std::chrono::high_resolution_clock::now();

    for (auto &left_tile : left_indexed.indexed_tensor) {
      std::fill(accumulator,
                accumulator + (left_inner_max) * (right_inner_max + 1), DT());
      for (const auto &left_entry : left_tile.second) {
        auto right_entry = right_indexed.indexed_tensor.find(left_entry.first);
        if (right_entry != right_indexed.indexed_tensor.end()) {
          for (auto &left_ev :
               left_entry.second) { // loop over (e_l, nnz_l): external
                                    // left, nnz at that external left.
            for (auto &right_ev : right_entry->second) {
              accumulator[left_ev.first * (right_inner_max + 1) +
                          right_ev.first] += left_ev.second * right_ev.second;
            }
          }
        }
      }
      // drain here.
      for (uint64_t i = 0; i < left_inner_max; i++) {
        for (uint64_t j = 0; j < right_inner_max + 1; j++) {
          if (accumulator[i * (right_inner_max + 1) + j] == DT())
            continue;
          uint64_t left_index =
              left_indexed.get_linear_index(left_tile.first, i);
          CompactCordinate this_cord =
              CompactCordinate(left_index, sample_left, j, sample_right);
          result_tensor.push_nnz(accumulator[i * (right_inner_max + 1) + j],
                                 this_cord);
        }
      }
    }
    end = std::chrono::high_resolution_clock::now();
    time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Time taken to contract: " << time_taken << std::endl;

    start = std::chrono::high_resolution_clock::now();

    end = std::chrono::high_resolution_clock::now();
    time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Time taken to writeback: " << time_taken << std::endl;
    std::cout << "Got " << result_tensor.get_nnz_count() << " nonzeros"
              << std::endl;
    return result_tensor;
  }

  
template <class AccType, class RES, class RIGHT>
  ListTensor<RES>
  fastcc_multiply(Tensor<RIGHT> &other, CoOrdinate left_contr,
                                 CoOrdinate right_contr, int tile_size = 100) {
    // for l_T
    //   for r_T
    //      for c
    //         for T_r
    //             for T_l
    int result_dimensionality =
        this->get_dimensionality() + other.get_dimensionality() -
        (left_contr.get_dimensionality() + right_contr.get_dimensionality());
    BoundedCoordinate sample_left = this->nonzeros[0]
                                      .get_coords()
                                      .remove(left_contr)
                                      .get_bounded(this->get_shape_ref());
    BoundedCoordinate sample_right = other.nonzeros[0]
                                         .get_coords()
                                         .remove(right_contr)
                                         .get_bounded(other.get_shape_ref());
    if(tile_size & (tile_size - 1)){
        tile_size = make_next_power_of_two(tile_size);
    }
    int num_workers = std::thread::hardware_concurrency();
    init_heaps(num_workers);
    // tf::Taskflow taskflow;
    // tf::Executor executor(num_workers);

    TileIndexedTensor<DT>* left_indexed = nullptr;
    TileIndexedTensor<DT>* right_indexed = nullptr;
    omp_set_nested(1);

    // -- 1. create hash tables for input tensors --
    auto stage_1_start = std::chrono::high_resolution_clock::now();
#pragma omp parallel num_threads(2)
    {
        if(omp_get_thread_num() == 0){
            left_indexed = new TileIndexedTensor<DT>(*this, left_contr, tile_size);
        }
        else{
            right_indexed = new TileIndexedTensor<DT>(other, right_contr, tile_size);
        }
    }
    auto stage_1_end = std::chrono::high_resolution_clock::now();
    auto stage_1_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(stage_1_end - stage_1_start)
            .count();
    std::cout << "Time taken to create hash tables for input tensors: " << stage_1_time << " ms" << std::endl;
    // -- 1 end --

    uint64_t left_inner_max = left_indexed->tile_size;
    uint64_t right_inner_max = right_indexed->tile_size;

    // -- 2. create thread local accumulators and result tensors --
    auto stage_2_start = std::chrono::high_resolution_clock::now();
    std::vector<AccType> thread_local_accumulators;

    ListTensor<RES>* thread_local_results = (ListTensor<RES>*) malloc(num_workers * sizeof(ListTensor<RES>));
    for (int _iter = 0; _iter < num_workers; _iter++) {
      thread_local_accumulators.push_back(
          AccType(left_inner_max, right_inner_max, _iter));
      thread_local_results[_iter] = ListTensor<RES>(result_dimensionality, _iter);
    }
    auto stage_2_end = std::chrono::high_resolution_clock::now();
    auto stage_2_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(stage_2_end - stage_2_start)
            .count();
    std::cout << "Time taken to create thread local accumulators: " << stage_2_time << " ms" << std::endl;
    // -- 2 end --

    // -- 3. create tasks: each task index a tile from left, a tile from right, and index the right elements and multiply, then update the thread local accumulator --
    std::chrono::milliseconds thread_local_hash_time[num_workers] {};
    std::chrono::milliseconds thread_local_accumulate_time[num_workers] {};
    std::chrono::milliseconds thread_local_drain_time[num_workers] {};
    std::chrono::milliseconds thread_local_total_time[num_workers] {};

    std::chrono::milliseconds tile_index_time(0);
    std::chrono::high_resolution_clock::time_point tile_index_start, tile_index_end;

    auto stage_3_start = std::chrono::high_resolution_clock::now();
    tile_index_start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for collapse(2) schedule(dynamic) num_threads(num_workers) 
    for (int i = 0; i < left_indexed->num_tiles(); i++){
      for (int j = 0; j < right_indexed->num_tiles(); j++){
          auto &left_tile = left_indexed->indexed_tensor[i];
          auto &right_tile = right_indexed->indexed_tensor[j];

          tile_index_end = std::chrono::high_resolution_clock::now();
          tile_index_time += std::chrono::duration_cast<std::chrono::milliseconds>(tile_index_end - tile_index_start);

          int my_id = omp_get_thread_num();
          AccType &myacc = thread_local_accumulators[my_id];
          myacc.reset_accumulator(i, j);

          std::chrono::high_resolution_clock::time_point hash_start, hash_end, acc_start, acc_end, drain_start, drain_end;
          std::chrono::milliseconds hash_time(0), acc_time(0), drain_time(0);
          // -- 3.1 index left and right tile --
          hash_start = std::chrono::high_resolution_clock::now();
          for (const auto &left_entry : left_tile) {
            auto right_entry = right_tile.find(left_entry.first);
            hash_end = std::chrono::high_resolution_clock::now();
            hash_time += std::chrono::duration_cast<std::chrono::milliseconds>(hash_end - hash_start);
            // -- 3.1 end --

            // -- 3.2 multiply and update accumulator --
            acc_start = std::chrono::high_resolution_clock::now();
            if (right_entry != right_tile.end()) {
              for (auto &left_ev :
                   left_entry.second) { // loop over (e_l, nnz_l): external
                                        // left, nnz at that external left.
                for (auto &right_ev : right_entry->second) {
                  uint64_t co_ordinate =
                      left_ev.first * right_inner_max + right_ev.first;
                  myacc.update(co_ordinate, left_ev.second * right_ev.second);
                }
              }
            }
            // -- 3.2 end --
            acc_end = std::chrono::high_resolution_clock::now();
            acc_time += std::chrono::duration_cast<std::chrono::milliseconds>(acc_end - acc_start);

            hash_start = std::chrono::high_resolution_clock::now();
          }

          // -- 3.3 drain accumulator to thread local result tensor --
          drain_start = std::chrono::high_resolution_clock::now();
          myacc.drain_into(thread_local_results[my_id],
                           sample_left, sample_right);
          drain_end = std::chrono::high_resolution_clock::now();
          drain_time += std::chrono::duration_cast<std::chrono::milliseconds>(drain_end - drain_start);

          thread_local_hash_time[my_id] += hash_time;
          thread_local_accumulate_time[my_id] += acc_time;
          thread_local_drain_time[my_id] += drain_time;

          thread_local_total_time[my_id] += (hash_time + acc_time + drain_time);

          tile_index_start = std::chrono::high_resolution_clock::now();
      }
    }

    auto stage_3_end = std::chrono::high_resolution_clock::now();
    auto stage_3_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(stage_3_end - stage_3_start)
            .count();

    for(int i = 0; i < num_workers; i++){
        std::cout << "Thread " << i << " hash time: " << thread_local_hash_time[i].count() << " ms" << std::endl;
        std::cout << "Thread " << i << " accumulate time: " << thread_local_accumulate_time[i].count() << " ms" << std::endl;
        std::cout << "Thread " << i << " drain time: " << thread_local_drain_time[i].count() << " ms" << std::endl;
        std::cout << "Thread " << i << " total time: " << thread_local_total_time[i].count() << " ms" << std::endl;
    }

    std::cout << "Tile indexing time: " << tile_index_time.count() << " ms" << std::endl;

    std::cout << "Time taken to compute all tiles: " << stage_3_time << " ms" << std::endl;

    // -- 3 end --

    // -- 4. merge thread local results --
    auto stage_4_start = std::chrono::high_resolution_clock::now();
    ListTensor<RES>& result_tensor = thread_local_results[0];
    for (int iter = 1; iter < num_workers; iter++) {
      result_tensor.concatenate(thread_local_results[iter]);
    }
    auto stage_4_end = std::chrono::high_resolution_clock::now();
    auto stage_4_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(stage_4_end - stage_4_start)
            .count();
    std::cout << "Time taken to merge thread local results: " << stage_4_time << " ms" << std::endl;
    // -- 4 end --

    return result_tensor;
  }

template <class RIGHT>
  uint64_t
  total_ht_keys(Tensor<RIGHT> &other, CoOrdinate left_contr,
               CoOrdinate right_contr, int left_tile_size = 100,
               int right_tile_size = 100) {
      uint64_t c_max = this->get_nonzeros()[0].get_coords().gather(left_contr).get_linearized_max();
    TileIndexedTensor<DT> *left_indexed;
    TileIndexedTensor<DT> *right_indexed;
#pragma omp parallel num_threads(2)
    {
      omp_set_nested(1);
      if (omp_get_thread_num() == 0) {
        left_indexed =
            new TileIndexedTensor<DT>(*this, left_contr, left_tile_size);
      } else {
        right_indexed =
            new TileIndexedTensor<DT>(other, right_contr, right_tile_size);
      }
    }
    uint64_t left_inner_max = left_indexed->tile_size;
    std::cout << "Left tile size is " << left_inner_max
              << ", number of tiles is " << left_indexed->num_tiles()
              << std::endl;
    uint64_t right_inner_max = right_indexed->tile_size;
    std::cout << "Right tile size is " << right_indexed->tile_size
              << ", number of tiles is " << right_indexed->num_tiles()
              << std::endl;
    // Number of queries is TotalActiveC_left(TL) * number of tiles on the right.
    //uint64_t num_keys = 0;
    //for(uint64_t tile_iter = 0; tile_iter < left_indexed->num_tiles(); tile_iter++){
    //    num_keys += left_indexed->num_active_columns(tile_iter);
    //}
    //for(uint64_t tile_iter = 0; tile_iter < right_indexed->num_tiles(); tile_iter++){
    //    num_keys += right_indexed->num_active_columns(tile_iter);
    //}
    return left_indexed->num_tiles() + right_indexed->num_tiles();
  }

  template <class RIGHT>
  std::pair<uint64_t, uint64_t>
  compute_cost(Tensor<RIGHT> &other, CoOrdinate left_contr,
               CoOrdinate right_contr, int left_tile_size = 100,
               int right_tile_size = 100) {
      uint64_t c_max = this->get_nonzeros()[0].get_coords().gather(left_contr).get_linearized_max();
    TileIndexedTensor<DT> *left_indexed;
    TileIndexedTensor<DT> *right_indexed;
#pragma omp parallel num_threads(2)
    {
      omp_set_nested(1);
      if (omp_get_thread_num() == 0) {
        left_indexed =
            new TileIndexedTensor<DT>(*this, left_contr, left_tile_size);
      } else {
        right_indexed =
            new TileIndexedTensor<DT>(other, right_contr, right_tile_size);
      }
    }
    uint64_t left_inner_max = left_indexed->tile_size;
    std::cout << "Left tile size is " << left_inner_max
              << ", number of tiles is " << left_indexed->num_tiles()
              << std::endl;
    uint64_t right_inner_max = right_indexed->tile_size;
    std::cout << "Right tile size is " << right_indexed->tile_size
              << ", number of tiles is " << right_indexed->num_tiles()
              << std::endl;
    // Number of queries is TotalActiveC_left(TL) * number of tiles on the right.
    uint64_t num_active_leftcols = 0;
    for(uint64_t tile_iter = 0; tile_iter < left_indexed->num_tiles(); tile_iter++){
        num_active_leftcols += left_indexed->num_active_columns(tile_iter);
    }
    uint64_t num_queries = num_active_leftcols * right_indexed->num_tiles();


    // Data volume of right tensor is summation_c(NNZCount_R(c) * FreqColumnC_L(TL, c)) TL is the tile size of left, FreqColumnC_L is the number of tiles in which c is active on the left tensor.
    uint64_t right_data_volume = 0;
    uint64_t left_data_volume = 0;
    std::cout<<"C_Max is "<<c_max<<std::endl;
    auto frequencies_left = left_indexed->idx_freq();
    auto frequencies_right = right_indexed->idx_freq();
    for(uint64_t c_iter = 0; c_iter < c_max; c_iter++){
        uint64_t nnz_count_R = right_indexed->nnz_in_idx_cord(c_iter);
        uint64_t nnz_count_L = left_indexed->nnz_in_idx_cord(c_iter);
        uint64_t freq_L = frequencies_left[c_iter];
        uint64_t freq_R = frequencies_right[c_iter];
        right_data_volume += nnz_count_R * freq_L;
        left_data_volume += nnz_count_L * freq_R;
    }
    return {num_queries, right_data_volume + left_data_volume};
  }

  template <class AccType, class RES, class RIGHT>
  ListTensor<RES>
  parallel_tile2d_outer_multiply(Tensor<RIGHT> &other, CoOrdinate left_contr,
                                 CoOrdinate right_contr, int left_tile_size = 100, int right_tile_size = 100) {
    // for l_T
    //   for r_T
    //      for c
    //         for T_r
    //             for T_l
    int result_dimensionality =
        this->get_dimensionality() + other.get_dimensionality() -
        (left_contr.get_dimensionality() + right_contr.get_dimensionality());
    BoundedCoordinate sample_left = this->nonzeros[0]
                                        .get_coords()
                                        .remove(left_contr)
                                        .get_bounded(this->get_shape_ref());
    BoundedCoordinate sample_right = other.nonzeros[0]
                                         .get_coords()
                                         .remove(right_contr)
                                         .get_bounded(other.get_shape_ref());
    std::cout << "Result dimensionality: " << result_dimensionality
              << std::endl;
    std::chrono::high_resolution_clock::time_point start, end;
    int num_workers = std::thread::hardware_concurrency() / 2;
    init_heaps(num_workers);
    tf::Taskflow taskflow;
    tf::Executor executor(num_workers);
    start = std::chrono::high_resolution_clock::now();

    TileIndexedTensor<DT>* left_indexed;
    TileIndexedTensor<DT>* right_indexed;
    #pragma omp parallel num_threads(2)
    {
    omp_set_nested(1);
        if(omp_get_thread_num() == 0){
            left_indexed = new TileIndexedTensor<DT>(*this, left_contr, left_tile_size);
        }
        else{
            right_indexed = new TileIndexedTensor<DT>(other, right_contr, right_tile_size);
        }

    }
    //TileIndexedTensor<DT> left_indexed =
    //    TileIndexedTensor<DT>(*this, left_contr, tile_size);
    uint64_t left_inner_max = left_indexed->tile_size;
    std::cout<<"Left tile size is "<<left_inner_max<<", number of tiles is "<<left_indexed->num_tiles()<<std::endl;
    //TileIndexedTensor<RIGHT> right_indexed =
    //    TileIndexedTensor<RIGHT>(other, right_contr, left_indexed.tile_size);
    uint64_t right_inner_max = right_indexed->tile_size;
    std::cout<<"Right tile size is "<<right_indexed->tile_size<<", number of tiles is "<<right_indexed->num_tiles()<<std::endl;

    std::vector<AccType> thread_local_accumulators;

    ListTensor<RES>* thread_local_results = (ListTensor<RES>*) malloc(num_workers * sizeof(ListTensor<RES>));
    for (int _iter = 0; _iter < num_workers; _iter++) {
      thread_local_accumulators.push_back(
          AccType(left_inner_max, right_inner_max, _iter));
      thread_local_results[_iter] = ListTensor<RES>(result_dimensionality, _iter);
    }
    Timer first_thread_timer;
    end = std::chrono::high_resolution_clock::now();

    double time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Time taken to index: " << time_taken << std::endl;
    start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < left_indexed->num_tiles(); i++) {
      for (int j = 0; j < right_indexed->num_tiles(); j++) {
            auto &left_tile = left_indexed->indexed_tensor[i];
            auto &right_tile = right_indexed->indexed_tensor[j];

            taskflow.emplace([&thread_local_accumulators, &thread_local_results,
                              i, j, &left_tile, &right_tile, &executor,
                              &first_thread_timer, &sample_left, &sample_right,
                              &right_inner_max, &left_inner_max]() mutable {
              int my_id = executor.this_worker_id();
              AccType &myacc = thread_local_accumulators[my_id];
              myacc.reset_accumulator(i, j);
              // if (my_id == 0) {
              //   first_thread_timer.start_timer("filling_tile");
              // }
              for (const auto &left_entry : left_tile) {
                auto right_entry = right_tile.find(left_entry.first);
                if (right_entry != right_tile.end()) {
                  for (auto &left_ev :
                       left_entry.second) { // loop over (e_l, nnz_l): external
                                            // left, nnz at that external left.
                    for (auto &right_ev : right_entry->second) {
                      uint64_t co_ordinate =
                          left_ev.first * right_inner_max + right_ev.first;
                      myacc.update(co_ordinate,
                                   left_ev.second * right_ev.second);
                    }
                  }
                }
              }
              // if (my_id == 0) {
              //   first_thread_timer.end_timer("filling_tile");
              // }
              if (my_id == 0) {
                first_thread_timer.start_timer("draining_tile");
              }
              // drain here.
              myacc.drain_into(thread_local_results[my_id], sample_left,
                               sample_right);
              if (my_id == 0) {
                first_thread_timer.end_timer("draining_tile");
              }
            });
      }
    }
    executor.run(taskflow).wait();
    end = std::chrono::high_resolution_clock::now();
    time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Time taken to contract: " << time_taken << std::endl;

    start = std::chrono::high_resolution_clock::now();
    ListTensor<RES>& result_tensor = thread_local_results[0];
    for (int iter = 1; iter < num_workers; iter++) {
      result_tensor.concatenate(thread_local_results[iter]);
    }
    end = std::chrono::high_resolution_clock::now();
    time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Time taken to writeback: " << time_taken << std::endl;
    first_thread_timer.print_all_times();
    //for (int iter = 0; iter < num_workers; iter++) {
    //  std::cout << "Thread " << iter << " times are:" << std::endl;
    //  // std::cout << "accumulator " << iter << " "
    //  //           << thread_local_accumulators[iter].percentage_saving()
    //  //           << "\% iterations saved" << std::endl;
    //}
    std::cout << "Got " << result_tensor.compute_nnz_count() << " nonzeros"
              << std::endl;
    return result_tensor;
  }

// Needs shape for left and right tensors
  // full outer multiplication
  template <class RIGHT>
  void microbench_outer_outer(Tensor<RIGHT> &other,
                                          CoOrdinate left_contr,
                                          CoOrdinate right_contr) {
      std::cout<<"microbench outer outer"<<std::endl;
    std::chrono::high_resolution_clock::time_point start, end;
    start = std::chrono::high_resolution_clock::now();
    SmallIndexedTensor<DT> left_indexed =
        SmallIndexedTensor<DT>(*this, left_contr);
    uint64_t left_inner_max = left_indexed.get_linearization_bound();
    SmallIndexedTensor<RIGHT> right_indexed =
        SmallIndexedTensor<RIGHT>(other, right_contr);
    uint64_t right_inner_max = right_indexed.get_linearization_bound();

    DT *accumulator =
        (DT *)malloc((left_inner_max + 1) * (right_inner_max + 1) * sizeof(DT));
    std::fill(accumulator,
              accumulator + (left_inner_max + 1) * (right_inner_max + 1), DT());
    if (accumulator == nullptr) {
      std::cerr << "Failed to allocate memory for accumulator" << std::endl;
      exit(1);
    } else {
      std::cout << "Allocated " << (left_inner_max + 1) * (right_inner_max + 1)
                << " elts for accumulator" << std::endl;
    }
    end = std::chrono::high_resolution_clock::now();
    double time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Time taken to index: " << time_taken << std::endl;
    uint64_t body_count = 0;
    start = std::chrono::high_resolution_clock::now();

    for (auto &left_entry : left_indexed.indexed_tensor) {
      auto right_entry = right_indexed.indexed_tensor.find(left_entry.first);
      if (right_entry != right_indexed.indexed_tensor.end()) {
        for (auto &left_ev :
             left_entry.second) { // loop over (e_l, nnz_l): external
                                  // left, nnz at that external left.
          for (auto &right_ev : right_entry->second) {
              body_count++;
          }
        }
      }
    }
    end = std::chrono::high_resolution_clock::now();
    time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Time taken to contract: " << time_taken << std::endl;
    std::cout<<"body was hit "<<body_count<<" times"<<std::endl;
    return;
  }

  template <class RIGHT>
  void microbench_tile2d(Tensor<RIGHT> &other, CoOrdinate left_contr,
                         CoOrdinate right_contr, int tile_size = 100) {
    // for l_T
    //   for r_T
    //      for c
    //         for T_r
    //             for T_l
      std::cout<<"microbench tile2d outer, tile size is "<<tile_size<<std::endl;

    std::chrono::high_resolution_clock::time_point start, end;
    start = std::chrono::high_resolution_clock::now();

    TileIndexedTensor<DT> left_indexed =
        TileIndexedTensor<DT>(*this, left_contr, tile_size);
    uint64_t left_inner_max = left_indexed.tile_size;
    TileIndexedTensor<RIGHT> right_indexed =
        TileIndexedTensor<RIGHT>(other, right_contr, left_indexed.tile_size);
    uint64_t right_inner_max = right_indexed.tile_size;

    end = std::chrono::high_resolution_clock::now();
    double time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Time taken to index: " << time_taken << std::endl;
    start = std::chrono::high_resolution_clock::now();

    uint64_t body_count = 0;
#pragma omp parallel for num_threads(8)
    for (auto left_tile = left_indexed.indexed_tensor.begin(); left_tile != left_indexed.indexed_tensor.end(); left_tile++) {
#pragma omp parallel for num_threads(8)
      for (auto right_tile = right_indexed.indexed_tensor.begin(); right_tile != right_indexed.indexed_tensor.end(); right_tile++) {

        for (const auto &left_entry : left_tile->second) {
          auto right_entry = right_tile->second.find(left_entry.first);
          if (right_entry != right_tile->second.end()) {
            for (auto &left_ev :
                 left_entry.second) { // loop over (e_l, nnz_l): external
                                      // left, nnz at that external left.
              for (auto &right_ev : right_entry->second) {
                int co_ordinate =
                    left_ev.first * right_inner_max + right_ev.first;
                body_count++;
              }
            }
          }
        }
      }
    }
    end = std::chrono::high_resolution_clock::now();
    time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Time taken to contract: " << time_taken << std::endl;
    std::cout << "body was hit " << body_count << " times" << std::endl;
  }

template <class RES, class RIGHT>
  ListTensor<RES>
  tile2d_outer_multiply(Tensor<RIGHT> &other, CoOrdinate left_contr,
                       CoOrdinate right_contr, int left_tile_size = 100, int right_tile_size = 100) {
    // for l_T
    //   for r_T
    //      for c
    //         for T_r
    //             for T_l
    int result_dimensionality =
        this->get_dimensionality() + other.get_dimensionality() -
        (left_contr.get_dimensionality() + right_contr.get_dimensionality());
    BoundedCoordinate sample_left = this->nonzeros[0]
                                        .get_coords()
                                        .remove(left_contr)
                                        .get_bounded(this->get_shape_ref());
    BoundedCoordinate sample_right = other.nonzeros[0]
                                         .get_coords()
                                         .remove(right_contr)
                                         .get_bounded(other.get_shape_ref());
    std::cout << "Result dimensionality: " << result_dimensionality
              << std::endl;
    std::chrono::high_resolution_clock::time_point start, end;
    std::cout<<"left tile size is "<<left_tile_size<<std::endl;
    init_heaps(1);
    start = std::chrono::high_resolution_clock::now();

    TileIndexedTensor<DT> left_indexed =
        TileIndexedTensor<DT>(*this, left_contr, left_tile_size);
    uint64_t left_inner_max = left_indexed.tile_size;
    TileIndexedTensor<RIGHT> right_indexed =
        TileIndexedTensor<RIGHT>(other, right_contr, right_tile_size);
    uint64_t right_inner_max = right_indexed.tile_size;
    std::cout<<"right tile size is "<<right_indexed.tile_size<<std::endl;

    TileAccumulator<RES> tile_accumulator(left_inner_max, right_inner_max);
    end = std::chrono::high_resolution_clock::now();
    double time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Time taken to index: " << time_taken << std::endl;
    ListTensor<RES> result_tensor(result_dimensionality);
    Timer mytimer;

    start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < left_indexed.num_tiles(); i++){
      for (int j = 0; j < right_indexed.num_tiles(); j++){
          auto &left_tile = left_indexed.indexed_tensor[i];
          auto &right_tile = right_indexed.indexed_tensor[j];
        tile_accumulator.reset_accumulator(i, j);
        for (const auto &left_entry : left_tile) {
          auto right_entry =
              right_tile.find(left_entry.first);
          mytimer.add_event("queries");
          if (right_entry != right_tile.end()) {
              mytimer.add_event("left_dv", left_entry.second.size());
              mytimer.add_event("right_dv", right_entry->second.size());
            for (auto &left_ev :
                 left_entry.second) { // loop over (e_l, nnz_l): external
                                      // left, nnz at that external left.
              for (auto &right_ev : right_entry->second) {
                tile_accumulator.update(left_ev.first * right_inner_max +
                                            right_ev.first,
                                        left_ev.second * right_ev.second);
              }
            }
          }
        }
        // drain here.
        //mytimer.start_timer("drain");
        tile_accumulator.drain_into(result_tensor, sample_left, sample_right);
        //mytimer.end_timer("drain");
      }
    }
    end = std::chrono::high_resolution_clock::now();
    time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Time taken to contract: " << time_taken << std::endl;
    mytimer.print_all_times();
    mytimer.print_all_events();

    start = std::chrono::high_resolution_clock::now();

    end = std::chrono::high_resolution_clock::now();
    time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Time taken to writeback: " << time_taken << std::endl;
    std::cout << "Got " << result_tensor.compute_nnz_count() << " nonzeros"
              << std::endl;
    return result_tensor;
  }

  // inner outer multiplication
  // has batch indices
  template <class RES, class RIGHT>
  CompactTensor<RES>
  _inner_outer_multiply(Tensor<RIGHT> &other, CoOrdinate left_contr,
                       CoOrdinate left_batch, CoOrdinate right_contr,
                       CoOrdinate right_batch) {
    // for l
    //    for coiter
    //       for r

    std::chrono::high_resolution_clock::time_point start, end;
    start = std::chrono::high_resolution_clock::now();
    CoOrdinate left_coiteration = CoOrdinate(left_batch, left_contr);
    // Make a vector with 0 to dimensionality - 1.
    std::vector<int> all_indices = std::vector<int>(this->get_dimensionality());
    std::iota(all_indices.begin(), all_indices.end(), 0);
    CoOrdinate left_external = CoOrdinate(all_indices).remove(left_coiteration);
    IndexedTensor<DT> left_indexed = IndexedTensor<DT>(*this, left_external);
    std::vector<int> batch_pos_afterhash(left_batch.get_dimensionality());
    std::iota(batch_pos_afterhash.begin(), batch_pos_afterhash.end(), 0);
    BoundedPosition batchpos = BoundedPosition(batch_pos_afterhash);

    CoOrdinate right_coiteration = CoOrdinate(right_batch, right_contr);
    IndexedTensor<RIGHT> right_indexed =
        IndexedTensor<RIGHT>(other, right_coiteration);
    end = std::chrono::high_resolution_clock::now();
    double time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Time taken to index: " << time_taken << std::endl;

    start = std::chrono::high_resolution_clock::now();
    OutputTensorHashMap3D<RES> result;
    for (auto &left_slice : left_indexed) {
      const BoundedCoordinate &left_ext_cordinate = left_slice.first;
      result.add_row(left_ext_cordinate);
      for (auto left_nnz : left_slice.second) {
        BoundedCoordinate batch_coord = left_nnz.first.gather(batchpos);
        auto right_slice = right_indexed.indexed_tensor.find(left_nnz.first);
        size_t size_hint = right_slice != right_indexed.indexed_tensor.end()
                               ? right_slice->second.size()
                               : 0;
        result.move_sliceptr(left_ext_cordinate, batch_coord, size_hint);
        if (right_slice != right_indexed.indexed_tensor.end()) {
          // There is atleast one nnz matching
          for (auto &right_nnz : right_slice->second) {
            const BoundedCoordinate &right_ext_cordinate = right_nnz.first;
            DT left_val = left_nnz.second;
            RIGHT right_val = right_nnz.second;
            RES outp;
            outp = left_val * right_val;
            result.update_last_row(right_ext_cordinate, outp);
          }
        }
      }
    }
    end = std::chrono::high_resolution_clock::now();
    time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Time taken to contract: " << time_taken << std::endl;
    std::cout << "Got " << result.get_nnz_count() << " nonzeros" << std::endl;
    start = std::chrono::high_resolution_clock::now();

    BoundedCoordinate sample_batch =
        left_indexed.begin()->second.begin()->first.gather(batchpos);
    BoundedCoordinate sample_left = left_indexed.begin()->first;
    BoundedCoordinate sample_right =
        right_indexed.begin()->second.begin()->first;
    CompactTensor<RES> result_tensor =
        result.drain(sample_batch, sample_left, sample_right);
    end = std::chrono::high_resolution_clock::now();
    time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Time taken to writeback: " << time_taken << std::endl;
    std::cout << "Got " << result_tensor.get_nnz_count() << " nonzeros"
              << std::endl;

    return result_tensor;
  }

  // This version has no batch indices
  template <class RES, class RIGHT>
  CompactTensor<RES>
  inner_outer_multiply(Tensor<RIGHT> &other, CoOrdinate left_contr,
                       CoOrdinate right_contr) {
    // for l
    //    for coiter
    //       for r

    std::chrono::high_resolution_clock::time_point start, end;
    start = std::chrono::high_resolution_clock::now();
    // Make a vector with 0 to dimensionality - 1.
    std::vector<int> all_indices = std::vector<int>(this->get_dimensionality());
    std::iota(all_indices.begin(), all_indices.end(), 0);
    CoOrdinate left_external = CoOrdinate(all_indices).remove(left_contr);
    SmallIndexedTensor<DT> left_indexed =
        SmallIndexedTensor<DT>(*this, left_external);

    SmallIndexedTensor<RIGHT> right_indexed =
        SmallIndexedTensor<RIGHT>(other, right_contr);
    BoundedCoordinate sample_left = this->nonzeros[0]
                                        .get_coords()
                                        .remove(left_contr)
                                        .get_bounded(this->get_shape_ref());
    BoundedCoordinate sample_right = other.nonzeros[0]
                                         .get_coords()
                                         .remove(right_contr)
                                         .get_bounded(other.get_shape_ref());
    end = std::chrono::high_resolution_clock::now();
    double time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Time taken to index: " << time_taken << std::endl;

    start = std::chrono::high_resolution_clock::now();
    OutputTensorHashMap2D<RES> result;
    for (auto &left_slice : left_indexed) {
      result.add_row(left_slice.first);
      for (auto left_nnz : left_slice.second) {
        auto right_slice = right_indexed.indexed_tensor.find(left_nnz.first);
        if (right_slice != right_indexed.indexed_tensor.end()) {
          // There is atleast one nnz matching
          for (auto &right_nnz : right_slice->second) {
            DT left_val = left_nnz.second;
            RIGHT right_val = right_nnz.second;
            RES outp;
            outp = left_val * right_val;
            result.update_last_row(right_nnz.first, outp);
          }
        }
      }
    }
    end = std::chrono::high_resolution_clock::now();
    time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Time taken to contract: " << time_taken << std::endl;
    std::cout << "Got " << result.get_nnz_count() << " nonzeros" << std::endl;
    start = std::chrono::high_resolution_clock::now();

    CompactTensor<RES> result_tensor =
        result.drain(sample_left, sample_right);
    end = std::chrono::high_resolution_clock::now();
    time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Time taken to writeback: " << time_taken << std::endl;
    std::cout << "Got " << result_tensor.get_nnz_count() << " nonzeros"
              << std::endl;

    return result_tensor;
  }

  // inner inner multiplication
  template <class RES, class RIGHT>
  ListTensor<RES> inner_inner_multiply(Tensor<RIGHT> &other,
                                       CoOrdinate left_contr,
                                       CoOrdinate right_contr) {
    // for l
    //    for r
    //       for coiter

    BoundedCoordinate sample_left =
        nonzeros[0].get_coords().remove(left_contr).get_bounded(shape);
    BoundedCoordinate sample_right = other.get_nonzeros()[0]
                                         .get_coords()
                                         .remove(right_contr)
                                         .get_bounded(other.get_shape_ref());
    std::chrono::high_resolution_clock::time_point start, end;
    start = std::chrono::high_resolution_clock::now();
    // Make a vector with 0 to dimensionality - 1.
    std::vector<int> all_indices = std::vector<int>(this->get_dimensionality());
    std::iota(all_indices.begin(), all_indices.end(), 0);
    CoOrdinate left_external = CoOrdinate(all_indices).remove(left_contr);
    SmallIndexedTensor<DT> left_indexed =
        SmallIndexedTensor<DT>(*this, left_external);

    CoOrdinate right_external = CoOrdinate(all_indices).remove(right_contr);
    SmallIndexedTensor<RIGHT> right_indexed =
        SmallIndexedTensor<RIGHT>(other, right_external);
    end = std::chrono::high_resolution_clock::now();
    double time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Time taken to index: " << time_taken << std::endl;
    // reserve largest possible space to begin with
    ListTensor<RES> result(
        this->get_dimensionality() + other.get_dimensionality() -
        (left_contr.get_dimensionality() + right_contr.get_dimensionality()));

    start = std::chrono::high_resolution_clock::now();
    for (auto liter = left_indexed.begin(); liter != left_indexed.end();
         liter++) {
      for (auto riter = right_indexed.begin(); riter != right_indexed.end();
           riter++) {
        std::vector<std::pair<uint64_t, DT>> &left = liter->second;
        std::vector<std::pair<uint64_t, DT>> &right = riter->second;
        DT res = sort_join(left, right);
        if (res == DT())
          continue;
        CompactCordinate this_cord = CompactCordinate(
            liter->first, sample_left, riter->first, sample_right);
        result.push_nnz(res, this_cord);
      }
    }
    end = std::chrono::high_resolution_clock::now();
    time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Time taken to contract: " << time_taken << std::endl;
    std::cout << "Got " << result.get_nnz_count() << " nonzeros" << std::endl;
    return result;
  }

  template <class RES, class RIGHT>
  AtomicListTensor<RES> parallel_inner_inner_multiply(Tensor<RIGHT> &other,
                                                      CoOrdinate left_contr,
                                                      CoOrdinate right_contr) {
    // for l
    //    for r
    //       for coiter

    BoundedCoordinate sample_left = nonzeros[0].get_coords().get_bounded(shape);
    BoundedCoordinate sample_right =
        other.get_nonzeros()[0].get_coords().get_bounded(other.get_shape_ref());
    std::chrono::high_resolution_clock::time_point start, end;
    start = std::chrono::high_resolution_clock::now();
    // Make a vector with 0 to dimensionality - 1.
    std::vector<int> all_indices = std::vector<int>(this->get_dimensionality());
    std::iota(all_indices.begin(), all_indices.end(), 0);
    CoOrdinate left_external = CoOrdinate(all_indices).remove(left_contr);
    SmallIndexedTensor<DT> left_indexed =
        SmallIndexedTensor<DT>(*this, left_external);

    CoOrdinate right_external = CoOrdinate(all_indices).remove(right_contr);
    SmallIndexedTensor<RIGHT> right_indexed =
        SmallIndexedTensor<RIGHT>(other, right_external);
    end = std::chrono::high_resolution_clock::now();
    double time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Time taken to index: " << time_taken << std::endl;
    // reserve largest possible space to begin with
    AtomicListTensor<RES> result(
        this->get_dimensionality() + other.get_dimensionality() -
        (left_contr.get_dimensionality() + right_contr.get_dimensionality()));
    tf::Executor exec;
    tf::Taskflow taskflow;
    start = std::chrono::high_resolution_clock::now();
    for (auto liter = left_indexed.begin(); liter != left_indexed.end();
         liter++) {
      for (auto riter = right_indexed.begin(); riter != right_indexed.end();
           riter++) {
        std::vector<std::pair<uint64_t, DT>> &left = liter->second;
        std::vector<std::pair<uint64_t, DT>> &right = riter->second;
        uint64_t leftcord = liter->first;
        uint64_t rightcord = riter->first;
        taskflow.emplace([&result, leftcord, rightcord, &sample_left,
                          &sample_right, left, right]() mutable {
          DT res = sort_join(left, right);
          if (res == DT())
            return;
          CompactCordinate this_cord =
              CompactCordinate(leftcord, sample_left, rightcord, sample_right);
          result.push_nnz(res, this_cord);
        });
      }
    }
    exec.run(taskflow).wait();
    end = std::chrono::high_resolution_clock::now();
    time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Size of left indexed " << left_indexed.get_size()
              << std::endl;
    std::cout << "Size of right indexed " << right_indexed.get_size()
              << std::endl;
    std::cout << "Time taken to contract: " << time_taken << std::endl;
    std::cout << "Got " << result.get_nnz_count() << " nonzeros" << std::endl;
    return result;
  }

  template <class RES, class RIGHT>
  CompactTensor<RES>
  parallel_inner_outer_multiply(Tensor<RIGHT> &other, CoOrdinate left_contr,
                                CoOrdinate left_batch, CoOrdinate right_contr,
                                CoOrdinate right_batch) {
    // for l
    //    for coiter
    //       for r

    std::chrono::high_resolution_clock::time_point start, end;
    start = std::chrono::high_resolution_clock::now();
    CoOrdinate left_coiteration = CoOrdinate(left_batch, left_contr);
    // Make a vector with 0 to dimensionality - 1.
    std::vector<int> all_indices = std::vector<int>(this->get_dimensionality());
    std::iota(all_indices.begin(), all_indices.end(), 0);
    CoOrdinate left_external = CoOrdinate(all_indices).remove(left_coiteration);
    IndexedTensor<DT> left_indexed = IndexedTensor<DT>(*this, left_external);
    std::vector<int> batch_pos_afterhash(left_batch.get_dimensionality());
    std::iota(batch_pos_afterhash.begin(), batch_pos_afterhash.end(), 0);
    BoundedPosition batchpos = BoundedPosition(batch_pos_afterhash);

    CoOrdinate right_coiteration = CoOrdinate(right_batch, right_contr);
    IndexedTensor<RIGHT> right_indexed =
        IndexedTensor<RIGHT>(other, right_coiteration);
    end = std::chrono::high_resolution_clock::now();
    double time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Time taken to index: " << time_taken << std::endl;

    tf::Executor exec;
    tf::Taskflow taskflow;
    std::forward_list<OutputTensorHashMap3D<RES>> task_local_results;
    start = std::chrono::high_resolution_clock::now();
    for (auto &left_slice : left_indexed) {
      // OutputTensorHashMap<RES> result;
      task_local_results.push_front(OutputTensorHashMap3D<RES>());
      OutputTensorHashMap3D<RES> &result = task_local_results.front();
      taskflow.emplace([&]() {
        const BoundedCoordinate &left_ext_cordinate = left_slice.first;
        result.add_row(left_ext_cordinate);
        for (auto left_nnz : left_slice.second) {
          BoundedCoordinate batch_coord = left_nnz.first.gather(batchpos);
          result.move_sliceptr(left_ext_cordinate, batch_coord);
          auto right_slice = right_indexed.indexed_tensor.find(left_nnz.first);
          if (right_slice != right_indexed.indexed_tensor.end()) {
            // There is atleast one nnz matching
            for (auto &right_nnz : right_slice->second) {
              const BoundedCoordinate &right_ext_cordinate = right_nnz.first;
              DT left_val = left_nnz.second;
              RIGHT right_val = right_nnz.second;
              RES outp;
              outp = left_val * right_val;
              result.update_last_row(right_ext_cordinate, outp);
            }
          }
        }
      });
    }
    exec.run(taskflow).wait();

    end = std::chrono::high_resolution_clock::now();
    time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Time taken to contract: " << time_taken << std::endl;

    BoundedCoordinate sample_batch =
        left_indexed.begin()->second.begin()->first.gather(batchpos);
    BoundedCoordinate sample_left = left_indexed.begin()->first;
    BoundedCoordinate sample_right =
        right_indexed.begin()->second.begin()->first;
    int total_nnz = 0;
    for (auto &res : task_local_results) {
      total_nnz += res.get_nnz_count();
    }
    std::cout << "Got " << total_nnz << " nonzeros" << std::endl;
    start = std::chrono::high_resolution_clock::now();
    CompactTensor<RES> result_tensor(total_nnz,
                                     sample_batch.get_dimensionality() +
                                         sample_left.get_dimensionality() +
                                         sample_right.get_dimensionality());
    int pos = 0;
    tf::Executor exec2;
    tf::Taskflow taskflow2;
    for (auto &res : task_local_results) {
      CompactTensor<RES> this_res =
          result_tensor.cut_at(pos, res.get_nnz_count());
      taskflow.emplace([res, this_res, &sample_batch, &sample_left,
                        &sample_right]() mutable {
        res.drain_into(this_res, sample_batch, sample_left, sample_right);
      });
      pos += res.get_nnz_count();
    }
    assert(pos == total_nnz);
    exec2.run(taskflow2).wait();
    end = std::chrono::high_resolution_clock::now();
    time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Time taken to writeback: " << time_taken << std::endl;

    return result_tensor;
  }

  // Very hacky in-place multiply, need to redo if it is really needed.
  template <class L, class R>
  void fill_values(Tensor<L> &left, Tensor<R> &right, CoOrdinate left_contr,
                   CoOrdinate left_batch, CoOrdinate right_contr,
                   CoOrdinate right_batch) {
    // make result tensor as the multiply of Tensor left and Tensor right
    Tensor<DT> result = left.template multiply<DT>(
        right, left_contr, left_batch, right_contr, right_batch);

    // TODO: remove this so that the next iteration can use the non-zero
    // positions.
    if (this->get_nonzeros().size() != 0) {
      this->delete_old_values();
    }
    for (auto &nnz : result) {
      this->get_nonzeros().push_back(nnz);
    }
  }

  template <class RIGHT>
  int count_ops(Tensor<RIGHT> &other, CoOrdinate left_contraction,
                CoOrdinate right_contraction) {
    assert(left_contraction.get_dimensionality() ==
           right_contraction.get_dimensionality());
    SymbolicTensor left = SymbolicTensor(*this);
    SymbolicTensor right = SymbolicTensor(other);
    return left.count_ops(right, left_contraction, right_contraction);
  }

  // In-place eltwise operations
  // For sparse tensors, the += can in-fact increase non-zeros.
#define OVERLOAD_OP(OP)                                                        \
  void operator OP(Tensor<DT> *other) {                                        \
    hashmap_vals indexed_tensor;                                               \
    for (auto &nnz : nonzeros) {                                               \
      indexed_tensor[nnz.get_coords()] = nnz.get_data();                       \
    }                                                                          \
    nonzeros.clear();                                                          \
    for (auto &nnz : (*other)) {                                               \
      auto ref = indexed_tensor.find(nnz.get_coords());                        \
      if (ref != indexed_tensor.end()) {                                       \
        ref.value() OP nnz.get_data();                                         \
      } else {                                                                 \
        nonzeros.push_back(nnz);                                               \
      }                                                                        \
    }                                                                          \
    for (auto &entry : indexed_tensor) {                                       \
      nonzeros.push_back(NNZ<DT>(entry.second, entry.first));                  \
    }                                                                          \
  }
  void operator+=(Tensor<DT> *other) {
    hashmap_vals indexed_tensor;
    for (auto &nnz : nonzeros) {
      indexed_tensor[nnz.get_coords()] = nnz.get_data();
    }
    nonzeros.clear();
    for (auto &nnz : other->get_nonzeros()) {
      auto ref = indexed_tensor.find(nnz.get_coords());
      if (ref != indexed_tensor.end()) {
        ref.value() += nnz.get_data();
      } else {
        nonzeros.push_back(nnz);
      }
    }
    for (auto &entry : indexed_tensor) {
      nonzeros.push_back(NNZ<DT>(entry.second, entry.first));
    }
  }
  OVERLOAD_OP(-=)
  OVERLOAD_OP(/=)
  DT operator[](CoOrdinate cord) {
    for (auto &nnz : nonzeros) {
      if (nnz.get_coords() == cord) {
        return nnz.get_data();
      }
    }
    return DT();
  }
};

template <class DT> Tensor<DT> CompactTensor<DT>::to_tensor() {
  Tensor<DT> result;
  for (int iter = 0; iter < num_nonzeros; iter++) {
    CompactNNZ<DT> *nnz = &nonzeros[iter];
    result.get_nonzeros().push_back(NNZ<DT>(
        nnz->get_data(), nnz->get_cord().as_coordinate(this->dimensionality)));
  }
  return result;
}
#endif
