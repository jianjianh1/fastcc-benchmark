#ifndef CONTRACT_HPP
#define CONTRACT_HPP
#include "coordinate.hpp"
#include "types.hpp"
#include <algorithm>
#include <boost/functional/hash.hpp>
#include <chrono>
#include <forward_list>
#include <iostream>
#include <random>
#include <ranges>
#include <tsl/hopscotch_map.h>
#include <tsl/hopscotch_set.h>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

template <typename T> static std::size_t hasharray(int size, T *arr) {
  return boost::hash_range(arr, arr + size);
}

template <class DT> class NNZ {
  DT data;
  CoOrdinate coords = CoOrdinate(0, nullptr);

public:
  // Constructor for a random value and coordinates
  NNZ(int dimensionality, int *shape) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> disf(0, 1);
    std::uniform_int_distribution<> dis_cords[dimensionality];
    for (int i = 0; i < dimensionality; i++) {
      dis_cords[i] = std::uniform_int_distribution<>(0, shape[i]);
    }
    std::vector<int> temp_coords;
    this->data = disf(gen);
    for (int i = 0; i < dimensionality; i++) {
      temp_coords.push_back(dis_cords[i](gen));
    }
    this->coords = CoOrdinate(temp_coords);
  }
  std::string to_string() const {
    std::string str = "";
    for (int i = 0; i < this->coords.get_dimensionality(); i++) {
      str += std::to_string(this->coords.get_index(i)) + " ";
    }
    str += std::to_string(data);
    return str;
  }
  int get_index(int dim) { return coords.get_index(dim); }
  DT get_data() { return data; }
  void set_zero();
  void operator+=(DT other) {
    data += other;
    if constexpr (std::is_class<DT>::value) {
      other.free();
    }
  }

  CoOrdinate &get_coords() { return coords; }

  // Constructor for a given value and coordinates
  NNZ(DT data, int dimensionality, int *coords)
      : data(data), coords(dimensionality, coords) {}
  NNZ(DT data, CoOrdinate coords) : data(data), coords(coords) {}
  NNZ(DT data, BoundedCoordinate bc) : data(data) {
    std::vector<int> vecords;
    for (int i = 0; i < bc.get_dimensionality(); i++) {
      vecords.push_back(bc.get_coordinate(i));
    }
    coords = CoOrdinate(vecords);
  }
  bool operator==(const NNZ &other) const {
    return data == other.data && coords == other.coords;
  }
};
template <> void NNZ<double>::set_zero() { data = 0.0; }
template <> void NNZ<float>::set_zero() { data = 0.0; }

template <class DT> class Tensor;
class SymbolicTensor {
  std::vector<CoOrdinate> indices;
  using hashmap_counts = tsl::hopscotch_map<CoOrdinate, int>;
  using hashmap_shape =
      tsl::hopscotch_map<CoOrdinate,
                         std::vector<std::pair<CoOrdinate, CoOrdinate>>>;
  std::vector<int> shape;
  void _infer_shape() {
    if (shape.size() == 0) {
      shape = std::vector<int>(indices[0].get_dimensionality(), -1);
      for (auto &cord : indices) {
        for (int i = 0; i < cord.get_dimensionality(); i++) {
          if (cord.get_index(i) > shape[i]) {
            shape[i] = cord.get_index(i);
          }
        }
      }
      for (int i = 0; i < shape.size(); i++) {
        shape[i] += 1;
      }
      std::vector<int> zeroindexed_shape;
      for (int i = 0; i < shape.size(); i++) {
        zeroindexed_shape.push_back(shape[i] - 1);
      }
      for (auto &ind : indices) {
        ind.set_shape(zeroindexed_shape);
      }
    }
  }

  void index_counts(CoOrdinate contraction, hashmap_counts &indexed_tensor) {
    for (auto &cord : *this) {
      auto filtered_coords = cord.gather(contraction);
      auto ref = indexed_tensor.find(filtered_coords);
      if (ref != indexed_tensor.end()) {
        ref.value() += 1;
      } else {
        indexed_tensor[filtered_coords] = 1;
      }
    }
    return;
  }

public:
  ~SymbolicTensor() { indices.clear(); }
  using iterator = typename std::vector<CoOrdinate>::iterator;
  iterator begin() { return indices.begin(); }
  iterator end() { return indices.end(); }
  template <class DT> SymbolicTensor(Tensor<DT> &some_tensor) {
    for (auto &nnz : some_tensor) {
      indices.push_back(nnz.get_coords());
    }
  }
  template <class It> SymbolicTensor(It begin, It end) {
    for (auto it = begin; it != end; it++) {
      indices.emplace_back(*it);
    }
  }
  SymbolicTensor(CoOrdinate singleton) { indices.push_back(singleton); }
  int get_size() { return indices.size(); }
  std::vector<int> get_shape() {
    _infer_shape();
    return shape;
  }
  int count_ops(SymbolicTensor &other, CoOrdinate left_contraction,
                CoOrdinate right_contraction) {
    assert(left_contraction.get_dimensionality() ==
           right_contraction.get_dimensionality());
    hashmap_counts first_index;
    this->index_counts(left_contraction, first_index);
    hashmap_counts second_index;
    other.index_counts(right_contraction, second_index);
    int ops = 0;
    for (auto &entry : first_index) {
      auto ref = second_index.find(entry.first);
      if (ref != second_index.end()) {
        ops += entry.second * ref->second;
      }
    }
    return ops;
  }

  void index_shape(CoOrdinate contraction, CoOrdinate batch,
                   hashmap_shape &indexed_tensor) {
    for (auto &cord : *this) {
      auto index_positions = CoOrdinate(batch, contraction);
      auto index_coords = cord.gather(index_positions);
      auto batch_coords = cord.gather(batch);
      auto external_coords = cord.remove(index_positions);
      auto ref = indexed_tensor.find(index_coords);
      if (ref != indexed_tensor.end()) {
        ref.value().push_back({batch_coords, external_coords});
      } else {
        indexed_tensor[index_coords] = {{batch_coords, external_coords}};
      }
    }
    return;
  }

  // Returns a set of coordinates that are the result of the contraction of two
  // tensors.
  // order is: (batch indices, left external indices, right external indices).
  // order within batch indices is dependent on the left operand
  tsl::hopscotch_set<CoOrdinate> output_shape(SymbolicTensor &other,
                                              CoOrdinate left_contr,
                                              CoOrdinate left_batch,
                                              CoOrdinate right_contr,
                                              CoOrdinate right_batch) {
    hashmap_shape first_index;
    this->index_shape(left_contr, left_batch, first_index);
    hashmap_shape second_index;
    other.index_shape(right_contr, right_batch, second_index);
    tsl::hopscotch_set<CoOrdinate> output;
    // Join on the keys of the map, cartesian product of the values.
    for (auto &entry : first_index) {
      auto ref = second_index.find(entry.first);
      if (ref != second_index.end()) {
        for (auto &leftcord : entry.second) {
          for (auto &rightcord : ref->second) {
            CoOrdinate batch_coords = leftcord.first;
            // CoOrdinate external_coords =
            //     CoOrdinate(leftcord.second, rightcord.second);
            CoOrdinate output_coords =
                CoOrdinate(batch_coords, leftcord.second, rightcord.second);
            output.insert(output_coords);
          }
        }
      }
    }
    return output;
  }

  std::pair<SymbolicTensor, double> contract_dense(SymbolicTensor &other,
                                                   CoOrdinate left_contraction,
                                                   CoOrdinate left_batch,
                                                   CoOrdinate right_contraction,
                                                   CoOrdinate right_batch) {
    CoOrdinate left_shape = this->get_shape();
    CoOrdinate right_shape = other.get_shape();
    CoOrdinate left_batch_shape = left_shape.gather(left_batch);
    CoOrdinate right_batch_shape = right_shape.gather(right_batch);
    CoOrdinate left_contraction_shape = left_shape.gather(left_contraction);
    CoOrdinate right_contraction_shape = right_shape.gather(right_contraction);

    assert(left_contraction_shape == right_contraction_shape);
    assert(left_batch_shape == right_batch_shape);
    CoOrdinate left_external_shape =
        left_shape.remove(CoOrdinate(left_batch, left_contraction));
    CoOrdinate right_external_shape =
        right_shape.remove(CoOrdinate(right_batch, right_contraction));
    CoOrdinate output_shape =
        CoOrdinate(left_batch_shape, left_external_shape, right_external_shape);
    double dense_cost = 1.0;
    for (auto &coord : output_shape) {
      dense_cost *= coord;
    }
    for (auto &coord : left_contraction_shape) {
      dense_cost *= coord;
    }
    std::vector<int> output_shape_vec;
    for (auto &_one_indexed_cord : output_shape) {
      output_shape_vec.push_back(_one_indexed_cord - 1);
    }
    SymbolicTensor output = SymbolicTensor(CoOrdinate(output_shape_vec));
    return {output, dense_cost};
  }

  std::pair<SymbolicTensor, double> contract(SymbolicTensor &other,
                                             CoOrdinate left_contraction,
                                             CoOrdinate left_batch,
                                             CoOrdinate right_contraction,
                                             CoOrdinate right_batch) {
    auto start = std::chrono::high_resolution_clock::now();
    tsl::hopscotch_set<CoOrdinate> output_coords = this->output_shape(
        other, left_contraction, left_batch, right_contraction, right_batch);
    auto end = std::chrono::high_resolution_clock::now();
    double time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    return {SymbolicTensor(output_coords.begin(), output_coords.end()),
            time_taken};
  }
};

template <class DT> class MultilevelIndexedTensor {
  using hashmap_vals =
      tsl::hopscotch_map<CoOrdinate, tsl::hopscotch_map<CoOrdinate, DT>>;
  using inner_map = tsl::hopscotch_map<CoOrdinate, DT>;
  hashmap_vals indexed_tensor;

public:
  MultilevelIndexedTensor(Tensor<DT> &base_tensor, CoOrdinate first_level_cords,
                          CoOrdinate second_level_cords) {
    assert(first_level_cords.get_dimensionality() +
               second_level_cords.get_dimensionality() ==
           base_tensor.get_dimensionality());
    for (auto &nnz : base_tensor) {
      auto first_level = nnz.get_coords().gather(first_level_cords);
      auto second_level = nnz.get_coords().gather(second_level_cords);
      auto it = indexed_tensor.find(first_level);
      if (it != indexed_tensor.end()) {
        it.value()[second_level] = nnz.get_data();
      } else {
        indexed_tensor[first_level] = {{second_level, nnz.get_data()}};
      }
    }
  }
  std::optional<DT> getif_valat(CoOrdinate first_level,
                                CoOrdinate second_level) {
    // lookups that fail in the first level will throw an exception
    inner_map &inner = indexed_tensor[first_level];
    auto it2 = inner.find(second_level);
    if (it2 != inner.end()) {
      return std::optional(it2.value());
    } else {
      return std::nullopt;
    }
  }
  inner_map &get_map(CoOrdinate first_level) {
    return indexed_tensor[first_level];
  }
};

template <class DT> class IndexedTensor {
  using hashmap_vals =
      tsl::hopscotch_map<BoundedCoordinate,
                         std::vector<std::pair<BoundedCoordinate, DT>>>;

public:
  hashmap_vals indexed_tensor;
  int *shape = nullptr;

  using iterator = typename hashmap_vals::iterator;
  using value_type = typename hashmap_vals::value_type;
  iterator begin() { return indexed_tensor.begin(); }
  iterator end() { return indexed_tensor.end(); }

  IndexedTensor(Tensor<DT> &base_tensor, CoOrdinate index_coords) {
    base_tensor._infer_shape();
    shape = base_tensor.get_shape_ref();
    BoundedPosition filter_pos =
        BoundedPosition(index_coords.begin(), index_coords.end());
    for (auto &nnz : base_tensor) {
      BoundedCoordinate full_coordinate = nnz.get_coords().get_bounded(shape);
      BoundedCoordinate index = full_coordinate.gather(filter_pos);
      BoundedCoordinate remaining = full_coordinate.remove(filter_pos);
      auto it = indexed_tensor.find(index);
      if (it != indexed_tensor.end()) {
        it.value().push_back({remaining, nnz.get_data()});
      } else {
        indexed_tensor[index] = {{remaining, nnz.get_data()}};
      }
    }
  }
  DT get_valat(CoOrdinate index_coords, CoOrdinate rem_coords) {
    BoundedCoordinate remaining_coords = rem_coords.get_bounded(shape);
    auto it = indexed_tensor.find(index_coords.get_bounded(shape));
    if (it != indexed_tensor.end()) {
      for (auto &pair : it.value()) {
        if (pair.first == remaining_coords) {
          return pair.second;
        }
      }
      std::cerr << "Found outer cord, can't find inner coordinate" << std::endl;
      exit(1);
    } else {
      std::cerr << "Can't even find outer coord" << index_coords.to_string()
                << std::endl;
      for (auto &cord : indexed_tensor) {
        std::cout << cord.first.to_string() << std::endl;
      }
      exit(1);
    }
  }

  // This is not exactly an equality operator, it just checks if the other
  // tensor has every value that this tensor has. Use it with a flip on the two
  // tensors to get equality operator. I'm lazy and didn't want to write a new
  // function.
  bool operator==(const IndexedTensor &other) const {
    if (indexed_tensor.size() != other.indexed_tensor.size()) {
      std::cerr << "Size mismatch, left was " << indexed_tensor.size()
                << ", right was " << other.indexed_tensor.size() << std::endl;
      return false;
    }
    for (auto &entry : indexed_tensor) {
      auto ref = other.indexed_tensor.find(entry.first);
      if (ref == other.indexed_tensor.end()) {
        std::cerr << "Index mismatch" << std::endl;
        return false;
      }
      if constexpr (std::is_class<DT>::value) {
        if (ref.value() != entry.second) {
          std::cerr << "Value mismatch" << std::endl;
          std::cerr << "Left: " << entry.second[0].second.to_string()
                    << std::endl;
          std::cerr << "Right: " << ref.value()[0].second.to_string()
                    << std::endl;
          return false;
        }
      } else {
#define FP_EQ(a, b) (fabs((a) - (b)) <= 5e-4)
        return FP_EQ(ref.value()[0].second, entry.second[0].second);
      }
    }
    return true;
  }
};

template <class DT> class OutputTensor {
private:
  using sparse_acc = tsl::hopscotch_map<OutputCoordinate, DT>;
  std::forward_list<std::pair<BoundedCoordinate, sparse_acc>> nonzeros;

public:
  bool is_same_row(BoundedCoordinate &left_ext) {
    if (this->nonzeros.empty())
      return true;
    if (this->nonzeros.front().first == left_ext)
      return true;
    return false;
  }
  void add_row(BoundedCoordinate left_ext) {
    nonzeros.push_front({left_ext, {}});
  }
  void update_last_row(OutputCoordinate& br, DT data) {
    sparse_acc& last_row = nonzeros.front().second;
    auto col_entry = last_row.find(br);
    if (col_entry != last_row.end()) {
      col_entry.value() += data;
    } else {
      last_row[br] = data;
    }
  }
  Tensor<DT> drain() {
    Tensor<DT> result;
    for (auto &row : nonzeros) {
      for (auto &nnz : row.second) {
        CoOrdinate batch = nnz.first.get_batch().as_coordinate();
        CoOrdinate leftex = row.first.as_coordinate();
        CoOrdinate rightex = nnz.first.get_right().as_coordinate();
        result.get_nonzeros().push_back(
            NNZ<DT>(nnz.second, CoOrdinate(batch, leftex, rightex)));
      }
    }
    return result;
  }
  int get_nnz_count() {
    return std::accumulate(
        nonzeros.begin(), nonzeros.end(), 0,
        [](int old_count, auto row) { return old_count + row.second.size(); });
  }
};

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
  // Constructor for a tensor of given shape and number of non-zeros, fills with
  // random values and indices
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

  // Returns a set of coordinates that are the result of the contraction of two
  // tensors.
  // order is: (batch indices, left external indices, right external indices).
  // order within batch indices is dependent on the left operand
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
                  OutputCoordinate output_coords = OutputCoordinate(
                      batch_coords, left_external, right_external);
                  RES outp;
                  if constexpr (std::is_same<DT, densevec>() &&
                                std::is_same<RIGHT, densevec>() &&
                                std::is_same<RES, densemat>()) {
                    outp = left_ev.second.densevec::outer(right_ev.second);
                  } else if constexpr (std::is_same<DT, densemat>() &&
                                       std::is_same<RIGHT, densemat>() &&
                                       std::is_same<RES, double>()) {
                    outp = left_ev.second.mult_reduce(right_ev.second);

                  } else {
                    outp = left_ev.second * right_ev.second;
                  }
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

  // inner outer multiplication
  template <class RES, class RIGHT>
  Tensor<RES> inner_outer_multiply(Tensor<RIGHT> &other, CoOrdinate left_contr,
                                   CoOrdinate left_batch,
                                   CoOrdinate right_contr,
                                   CoOrdinate right_batch) {
    // for l
    //    for coiter
    //       for r

    std::chrono::high_resolution_clock::time_point start, end;
    start = std::chrono::high_resolution_clock::now();
    this->_infer_dimensionality();
    this->_infer_shape();
    other._infer_dimensionality();
    other._infer_shape();
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

    // tsl::hopscotch_map<OutputCoordinate, RES> result;
    start = std::chrono::high_resolution_clock::now();
    OutputTensor<RES> result;
    for (auto &left_slice : left_indexed) {
      BoundedCoordinate left_ext_cordinate = left_slice.first;
      result.add_row(left_ext_cordinate);
      for (auto left_nnz : left_slice.second) {
              BoundedCoordinate batch_coord = left_nnz.first.gather(batchpos);
              auto right_slice =
                  right_indexed.indexed_tensor.find(left_nnz.first);
              if (right_slice != right_indexed.indexed_tensor.end()) {
                // There is atleast one nnz matching
                for (auto &right_nnz : right_slice->second) {
                  BoundedCoordinate right_ext_cordinate = right_nnz.first;
                  DT left_val = left_nnz.second;
                  RIGHT right_val = right_nnz.second;
                  RES outp;
                  if constexpr (std::is_same<DT, densevec>() &&
                                std::is_same<RIGHT, densevec>() &&
                                std::is_same<RES, densemat>()) {
                    outp = left_val.densevec::outer(right_val);
                  } else if constexpr (std::is_same<DT, densemat>() &&
                                       std::is_same<RIGHT, densemat>() &&
                                       std::is_same<RES, double>()) {
                    outp = left_val.mult_reduce(right_val);

                  } else {
                    outp = left_val * right_val;
                  }
                  OutputCoordinate output_coords = OutputCoordinate(
                      batch_coord, BoundedCoordinate(), right_ext_cordinate);
                  result.update_last_row(output_coords, outp);
                  //result.is_same_row(left_ext_cordinate); //TODO: remove before flight
                  // auto result_ref = result.find(output_coords);
                  // if (result_ref != result.end()) {
                  //   result_ref.value() += outp;
                  // } else {
                  //   result[output_coords] = outp;
                  // }
                }
              }
      }
    }
    end = std::chrono::high_resolution_clock::now();
    time_taken =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    std::cout << "Time taken to contract: " << time_taken << std::endl;
    //std::cout<<"Got "<<result.get_nnz_count()<<" nonzeros"<<std::endl;

    Tensor<RES> result_tensor = result.drain();

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
#endif
