#ifndef CONTRACT_HPP
#define CONTRACT_HPP
#include "types.hpp"
#include <algorithm>
#include <boost/functional/hash.hpp>
#include <iostream>
#include <chrono>
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

class CoOrdinate {
#define BITWIDTH (256)
  std::vector<int> coords;
  std::bitset<BITWIDTH> mybits;

public:
  using iterator = typename std::vector<int>::iterator;
  using const_iterator = typename std::vector<int>::const_iterator;
  using value_type = typename std::vector<int>::value_type;
  // iterator begin() { return coords.begin(); }
  iterator begin() { return coords.begin(); }
  const_iterator begin() const { return coords.begin(); }
  iterator end() { return coords.end(); }
  const_iterator end() const { return coords.end(); }
  CoOrdinate(int dimensionality, int *coords) {
    for (int i = 0; i < dimensionality; i++) {
      this->coords.push_back(coords[i]);
    }
    for (auto &cord : this->coords) {
      mybits <<= (sizeof(int)*8);
      mybits |= std::bitset<BITWIDTH>(cord);
    }
  }
  CoOrdinate(std::vector<int> data) {
    this->coords = data;
    for (auto &cord : this->coords) {
      mybits <<= (sizeof(int)*8);
      mybits |= std::bitset<BITWIDTH>(cord);
    }
  }
  std::string to_string() const;
  void write(std::string filename) const;

  // This is going to concatenate two coordinates
  CoOrdinate(CoOrdinate left, CoOrdinate right) {
      coords.reserve(left.get_dimensionality() + right.get_dimensionality());
      //memcpy(coords.data(), left.coords.data(), left.get_dimensionality() * sizeof(int));
      //memcpy(coords.data() + left.get_dimensionality(), right.coords.data(), right.get_dimensionality() * sizeof(int));
    coords.insert(coords.end(), left.coords.begin(), left.coords.end());
    coords.insert(coords.end(), right.coords.begin(), right.coords.end());
    for (auto &cord : this->coords) {
      mybits <<= (sizeof(int)*8); // sizeof is in bytes, so we need to multiply by 8 to get bits
      mybits |= std::bitset<BITWIDTH>(cord);
    }
  }

  CoOrdinate(CoOrdinate left, CoOrdinate mid, CoOrdinate right){
      coords.reserve(left.get_dimensionality() + mid.get_dimensionality() + right.get_dimensionality());
      coords.insert(coords.end(), left.coords.begin(), left.coords.end());
      coords.insert(coords.end(), mid.coords.begin(), mid.coords.end());
      coords.insert(coords.end(), right.coords.begin(), right.coords.end());
      for (auto &cord : this->coords) {
        mybits <<= (sizeof(int)*8); // sizeof is in bytes, so we need to multiply by 8 to get bits
        mybits |= std::bitset<BITWIDTH>(cord);
      }
  }

  CoOrdinate gather(CoOrdinate positions) const{
    // TODO remove before flight
    if (positions.get_dimensionality() > this->get_dimensionality()) {
      std::cout << "Error, trying to gather more dimensions than there are in "
                   "the tensor"
                << std::endl;
      std::cout << "positions asked: " << positions.to_string() << std::endl;
      std::cout << "Gathered: " << positions.get_dimensionality()
                << " Tensor: " << this->get_dimensionality() << std::endl;
    }
    assert(positions.get_dimensionality() <= this->get_dimensionality());
    // TODO remove before flight
    for (auto &cord : positions) {
      if (cord >= this->get_dimensionality()) {
        std::cout << "Error, trying to gather a coordinate that doesn't exist"
                  << std::endl;
        std::cout << "Asked for " << cord << " in a tensor of dimensionality "
                  << this->get_dimensionality() << std::endl;
        exit(1);
      }
    }
    std::vector<int> gathered;
    for (int i = 0; i < positions.get_dimensionality(); i++) {
      gathered.push_back(coords[positions.get_index(i)]);
    }
    return gathered;
  }

  CoOrdinate remove(CoOrdinate positions) {
    std::vector<int> removed;
    for (int i = 0; i < this->get_dimensionality(); i++) {
      if (std::find(positions.begin(), positions.end(), i) == positions.end()) {
        removed.push_back(coords[i]);
      }
    }
    return removed;
  }

  int get_index(int dim) const { return coords[dim]; }
  int get_dimensionality() const { return coords.size(); }
  std::bitset<BITWIDTH> get_bits() const { return mybits; }
  bool operator==(const CoOrdinate &other) const {
    return mybits == other.mybits;
    // if (this->get_dimensionality() != other.get_dimensionality()) {
    //   return false;
    // }

    // for (int i = 0; i < this->get_dimensionality(); i++) {
    //   if (coords[i] != other.coords[i]) {
    //     return false;
    //   }
    // }
    // return true;
  }
};

template <> struct std::hash<CoOrdinate> {
  std::size_t operator()(const CoOrdinate &c) const {
    //std::string catted_cord = "";
    //for (auto &&coord : c) {
    //  catted_cord += std::to_string(coord);
    //  catted_cord += ",";
    //}
    //return std::hash<std::string>{}(catted_cord);
    return std::hash<std::bitset<BITWIDTH>>{}(c.get_bits());
  }
};

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

  CoOrdinate get_coords() { return coords; }

  // Constructor for a given value and coordinates
  NNZ(DT data, int dimensionality, int *coords)
      : data(data), coords(dimensionality, coords) {}
  NNZ(DT data, CoOrdinate coords) : data(data), coords(coords) {}
  bool operator==(const NNZ &other) const {
    return data == other.data && coords == other.coords;
  }
};

template <class DT> class Tensor;
class SymbolicTensor {
  std::vector<CoOrdinate> indices;
  using hashmap_counts = tsl::hopscotch_map<CoOrdinate, int>;
  using hashmap_shape =
      tsl::hopscotch_map<CoOrdinate,
                         std::vector<std::pair<CoOrdinate, CoOrdinate>>>;
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
  int get_size() { return indices.size(); }
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
            //CoOrdinate external_coords =
            //    CoOrdinate(leftcord.second, rightcord.second);
            CoOrdinate output_coords =
                CoOrdinate(batch_coords, leftcord.second, rightcord.second);
            output.insert(output_coords);
          }
        }
      }
    }
    return output;
  }

  std::pair<SymbolicTensor, double> contract(SymbolicTensor &other, CoOrdinate left_contraction,
                          CoOrdinate left_batch, CoOrdinate right_contraction,
                          CoOrdinate right_batch) {
      auto start = std::chrono::high_resolution_clock::now();
      tsl::hopscotch_set<CoOrdinate> output_coords = this->output_shape(
        other, left_contraction, left_batch, right_contraction, right_batch);
    auto end = std::chrono::high_resolution_clock::now();
    double time_taken =  std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return {SymbolicTensor(output_coords.begin(), output_coords.end()), time_taken};
  }
};
template <class DT> class IndexedTensor {
  using hashmap_vals =
      tsl::hopscotch_map<CoOrdinate, std::vector<std::pair<CoOrdinate, DT>>>;

public:
  hashmap_vals indexed_tensor;

  IndexedTensor(Tensor<DT> &base_tensor, CoOrdinate index_coords) {
    for (auto &nnz : base_tensor) {
      auto it = indexed_tensor.find(nnz.get_coords().gather(index_coords));
      if (it != indexed_tensor.end()) {
        it.value().push_back(
            {nnz.get_coords().remove(index_coords), nnz.get_data()});
      } else {
        indexed_tensor[nnz.get_coords().gather(index_coords)] = {
            {nnz.get_coords().remove(index_coords), nnz.get_data()}};
      }
    }
  }
  DT get_valat(CoOrdinate index_coords, CoOrdinate remaining_coords) {
    auto it = indexed_tensor.find(index_coords);
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
      if (ref.value() != entry.second) {
        std::cerr << "Value mismatch" << std::endl;
        return false;
      }
    }
    return true;
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
  int get_dimensionality() { return dimensionality; }
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
      //TODO: might be a leak....but need to replace with smart pointers maybe.
    //if constexpr (std::is_class<DT>::value) {
    //  for (auto &nnz : nonzeros) {
    //    nnz.get_data().free();
    //  }
    //}
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

  Tensor contract(Tensor &other, CoOrdinate left_contraction,
                  CoOrdinate left_batch, CoOrdinate right_contraction,
                  CoOrdinate right_batch) {
    tsl::hopscotch_set<CoOrdinate> output_coords = this->output_shape(
        other, left_contraction, left_batch, right_contraction, right_batch);
    return Tensor(output_coords.begin(), output_coords.end());
  }

  template <class RES, class RIGHT>
  Tensor<RES> multiply(Tensor<RIGHT> &other, CoOrdinate left_contr,
                       CoOrdinate left_batch, CoOrdinate right_contr,
                       CoOrdinate right_batch) {
    CoOrdinate left_idx_pos = CoOrdinate(left_batch, left_contr);
    IndexedTensor<DT> left_indexed = IndexedTensor<DT>(*this, left_idx_pos);
    CoOrdinate right_idx_pos = CoOrdinate(right_batch, right_contr);
    IndexedTensor<RIGHT> right_indexed =
        IndexedTensor<RIGHT>(other, right_idx_pos);

    tsl::hopscotch_map<CoOrdinate, RES> result;

    std::vector<int> batch_pos_afterhash(left_batch.get_dimensionality());
    std::iota(batch_pos_afterhash.begin(), batch_pos_afterhash.end(), 0);
    CoOrdinate idx_batch_pos = CoOrdinate(batch_pos_afterhash);

    for (auto &left_entry : left_indexed.indexed_tensor) {
      auto right_entry = right_indexed.indexed_tensor.find(left_entry.first);
      if (right_entry != right_indexed.indexed_tensor.end()) {
        for (auto &left_ev :
             left_entry.second) { // loop over (e_l, nnz_l): external left, nnz
                                  // at that external left.
          for (auto &right_ev : right_entry->second) {
            CoOrdinate batch_coords = left_entry.first.gather(
                batch_pos_afterhash); // assumes that batch positions are leftmost, so
                             // they will work with a left subset.
            //CoOrdinate external_coords = CoOrdinate(
            //    left_ev.first,
            //    right_ev.first); // convention to put left followed by right
            CoOrdinate output_coords =
                CoOrdinate(batch_coords, left_ev.first, right_ev.first);
            RES outp;
            if constexpr (std::is_same<DT, densevec>() &&
                          std::is_same<RIGHT, densevec>() &&
                          std::is_same<RES, densemat>()) {
              outp = left_ev.second.densevec::outer(right_ev.second);
            } else if constexpr(std::is_same<DT, densemat>() && std::is_same<RIGHT, densemat>() && std::is_same<RES, double>()){
                outp = left_ev.second.mult_reduce(right_ev.second);

            }else {
              outp = left_ev.second * right_ev.second;
            }
            auto result_ref = result.find(output_coords);
            if (result_ref != result.end()) {
              result_ref.value() += outp;
            } else {
              result[output_coords] = outp;
            }
          }
        }
      }
    }
    Tensor<RES> result_tensor(result.size());

    for (auto nnz : result) {
      result_tensor.get_nonzeros().push_back(NNZ<RES>(nnz.second, nnz.first));
    }
    return result_tensor;
  }

  template <class L, class R>
  void fill_values(Tensor<L> &left, Tensor<R> &right, CoOrdinate left_contr,
                   CoOrdinate left_batch, CoOrdinate right_contr,
                   CoOrdinate right_batch) {
      //TODO: refactor this above
    CoOrdinate left_idx_pos = CoOrdinate(left_batch, left_contr);
    IndexedTensor<L> left_indexed = IndexedTensor<L>(left, left_idx_pos);
    CoOrdinate right_idx_pos = CoOrdinate(right_batch, right_contr);
    IndexedTensor<R> right_indexed =
        IndexedTensor<R>(right, right_idx_pos);

    tsl::hopscotch_map<CoOrdinate, DT> result;

    std::vector<int> batch_pos_afterhash(left_batch.get_dimensionality());
    std::iota(batch_pos_afterhash.begin(), batch_pos_afterhash.end(), 0);
    CoOrdinate idx_batch_pos = CoOrdinate(batch_pos_afterhash);

    for (auto &left_entry : left_indexed.indexed_tensor) {
      auto right_entry = right_indexed.indexed_tensor.find(left_entry.first);
      if (right_entry != right_indexed.indexed_tensor.end()) {
        for (auto &left_ev :
             left_entry.second) { // loop over (e_l, nnz_l): external left, nnz
                                  // at that external left.
          for (auto &right_ev : right_entry->second) {
            CoOrdinate batch_coords = left_entry.first.gather(
                batch_pos_afterhash); // assumes that batch positions are leftmost, so
                             // they will work with a left subset.
            //CoOrdinate external_coords = CoOrdinate(
            //    left_ev.first,
            //    right_ev.first); // convention to put left followed by right
            CoOrdinate output_coords =
                CoOrdinate(batch_coords, left_ev.first, right_ev.first);
            DT outp;
            if constexpr (std::is_same<L, densevec>() &&
                          std::is_same<R, densevec>() &&
                          std::is_same<DT, densemat>()) {
              outp = left_ev.second.densevec::outer(right_ev.second);
            } else if constexpr (std::is_same<L, densemat>() &&
                                 std::is_same<R, densemat>() &&
                                 std::is_same<DT, double>()) {
              outp = left_ev.second.mult_reduce(right_ev.second);

            }

            else {
              outp = left_ev.second * right_ev.second;
            }
            auto result_ref = result.find(output_coords);
            if (result_ref != result.end()) {
              result_ref.value() += outp;
            } else {
              result[output_coords] = outp;
            }
          }
        }
      }
    }

    if (this->get_nonzeros().size() != 0) {
      this->delete_old_values();
    }
    for (auto &nnz : result) {
      this->get_nonzeros().push_back(NNZ<DT>(nnz.second, nnz.first));
    }
    result.clear();
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
    for (auto &nnz : (*other)) {                                                  \
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
