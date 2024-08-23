#ifndef CONTRACT_HPP
#define CONTRACT_HPP
#include "types.hpp"
#include <algorithm>
#include <boost/functional/hash.hpp>
#include <iostream>
#include <random>
#include <ranges>
#include <tsl/hopscotch_map.h>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
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
      mybits <<= sizeof(int);
      mybits |= std::bitset<BITWIDTH>(cord);
    }
  }
  CoOrdinate(std::vector<int> data) {
    this->coords = data;
    for (auto &cord : this->coords) {
      mybits <<= sizeof(int);
      mybits |= std::bitset<BITWIDTH>(cord);
    }
  }
  std::string to_string() const;
  void write(std::string filename) const;

  // This is going to concatenate two coordinates
  CoOrdinate(CoOrdinate left, CoOrdinate right) {
    coords.insert(coords.end(), left.coords.begin(), left.coords.end());
    coords.insert(coords.end(), right.coords.begin(), right.coords.end());
    for (auto &cord : this->coords) {
      mybits <<= sizeof(int);
      mybits |= std::bitset<BITWIDTH>(cord);
    }
  }

  CoOrdinate gather(CoOrdinate positions) {
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
    std::string catted_cord = "";
    for (auto &&coord : c) {
      catted_cord += std::to_string(coord);
      catted_cord += ",";
    }
    return std::hash<std::string>{}(catted_cord);
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
  std::unordered_set<CoOrdinate> output_shape(SymbolicTensor &other,
                                              CoOrdinate left_contr,
                                              CoOrdinate left_batch,
                                              CoOrdinate right_contr,
                                              CoOrdinate right_batch) {
    hashmap_shape first_index;
    this->index_shape(left_contr, left_batch, first_index);
    hashmap_shape second_index;
    other.index_shape(right_contr, right_batch, second_index);
    std::unordered_set<CoOrdinate> output;
    // Join on the keys of the map, cartesian product of the values.
    for (auto &entry : first_index) {
      auto ref = second_index.find(entry.first);
      if (ref != second_index.end()) {
        for (auto &leftcord : entry.second) {
          for (auto &rightcord : ref->second) {
            CoOrdinate batch_coords = leftcord.first;
            CoOrdinate external_coords =
                CoOrdinate(leftcord.second, rightcord.second);
            CoOrdinate output_coords =
                CoOrdinate(batch_coords, external_coords);
            output.insert(output_coords);
          }
        }
      }
    }
    return output;
  }

  SymbolicTensor contract(SymbolicTensor &other, CoOrdinate left_contraction,
                          CoOrdinate left_batch, CoOrdinate right_contraction,
                          CoOrdinate right_batch) {
    std::unordered_set<CoOrdinate> output_coords = this->output_shape(
        other, left_contraction, left_batch, right_contraction, right_batch);
    return SymbolicTensor(output_coords.begin(), output_coords.end());
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
      // TODO: replace this linear scan with a hash lookup. hotspot here.
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
    if constexpr (std::is_class<DT>::value) {
      for (auto &nnz : nonzeros) {
        nnz.get_data().free();
      }
    }
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
  Tensor(int size) { nonzeros.reserve(size); }
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
  std::unordered_set<CoOrdinate>
  output_shape(Tensor &other, CoOrdinate left_contr, CoOrdinate left_batch,
               CoOrdinate right_contr, CoOrdinate right_batch) {
    auto right_symbolic = SymbolicTensor(other);
    return SymbolicTensor(*this).output_shape(
        right_symbolic, left_contr, left_batch, right_contr, right_batch);
  }

  Tensor contract(Tensor &other, CoOrdinate left_contraction,
                  CoOrdinate left_batch, CoOrdinate right_contraction,
                  CoOrdinate right_batch) {
    std::unordered_set<CoOrdinate> output_coords = this->output_shape(
        other, left_contraction, left_batch, right_contraction, right_batch);
    return Tensor(output_coords.begin(), output_coords.end());
  }

  template <class RES, class RIGHT>
  Tensor<RES> multiply(Tensor<RIGHT> &other, CoOrdinate left_contr,
                       CoOrdinate left_batch, CoOrdinate right_contr,
                       CoOrdinate right_batch) {
    // Store the output as a hashmap of coordinates to values
    // Co-iterate over the left and right op just like we did for op-counting,
    // just add and update the value if it exists. We need a way to get a
    // default value from the abstract datatype. We also need to write functions
    // to accumulate (+=) in-place with another object of the same abstract
    // type.
    hashmap_shape first_index;
    auto sym_this = SymbolicTensor(*this);
    sym_this.index_shape(left_contr, left_batch, first_index);
    // for(auto &nnz: other){
    //   std::cout<<"Base tensor keys
    //   "<<nnz.get_coords().to_string()<<std::endl;
    // }
    // std::cout<<"Right contraction "<<right_contr.to_string()<<std::endl;
    // std::cout<<"Right batch "<<right_batch.to_string()<<std::endl;
    hashmap_shape second_index;
    auto sym_other = SymbolicTensor(other);
    sym_other.index_shape(right_contr, right_batch, second_index);
    // for(auto &cord: second_index){
    //   std::cout<<"Sym tensor keys "<<cord.first.to_string()<<std::endl;
    // }
    auto left_indcords = CoOrdinate(left_batch, left_contr);
    IndexedTensor<DT> this_indexed(*this, left_indcords);
    auto right_indcords = CoOrdinate(right_batch, right_contr);
    IndexedTensor<RIGHT> other_indexed(other, right_indcords);
    // for(auto &cord: other_indexed.indexed_tensor){
    //   std::cout<<"IndexedTensor keys "<<cord.first.to_string()<<std::endl;
    // }
    std::unordered_map<CoOrdinate, RES> output;
    // std::cout<<"Starting to form output hashmap, gonna coiterate"<<std::endl;
    //  Join on the keys of the map, cartesian product of the values.
    for (auto &entry : first_index) {
      auto ref = second_index.find(entry.first);
      if (ref != second_index.end() && second_index.size() > 0) {
        // std::cout<<"Found a match "<<entry.first.to_string()<<std::endl;
        for (auto &leftcord : entry.second) {
          for (auto &rightcord : ref->second) {
            CoOrdinate batch_coords = leftcord.first;
            CoOrdinate external_coords =
                CoOrdinate(leftcord.second, rightcord.second);
            CoOrdinate output_coords =
                CoOrdinate(batch_coords, external_coords);
            auto ref = output.find(output_coords);
            if (ref != output.end() && output.size() > 0) {
              // This is a weird behaviour where if the hashmap is empty, we get
              // a dangling iterator that causes a segfault.
              RES outp;
              if constexpr (std::is_same<DT, densevec>() &&
                            std::is_same<RIGHT, densevec>() &&
                            std::is_same<RES, densemat>()) {
                outp = this_indexed.get_valat(entry.first, leftcord.second)
                           .densevec::outer(other_indexed.get_valat(
                               entry.first, rightcord.second));
              } else {
                outp = this_indexed.get_valat(entry.first, leftcord.second) *
                       other_indexed.get_valat(entry.first, rightcord.second);
              }
              ref->second += outp;
            } else {

              // if constexpr (std::is_class<DT>::value) {
              //   std::cout << "Left vector "
              //             << this->get_valat(leftcord.first).to_string()
              //             << std::endl;
              // } else {
              //   std::cout << "Left vector " <<
              //   this->get_valat(rightcord.first)
              //             << std::endl;
              // }
              // if constexpr (std::is_class<RIGHT>::value) {
              //   std::cout << "Right vector "
              //             << other.get_valat(leftcord.first).to_string()
              //             << std::endl;
              // } else {
              //   std::cout << "Right vector " <<
              //   other.get_valat(rightcord.first)
              //             << std::endl;
              // }
              RES outp;
              if constexpr (std::is_same<DT, densevec>() &&
                            std::is_same<RIGHT, densevec>() &&
                            std::is_same<RES, densemat>()) {
                outp = this_indexed.get_valat(entry.first, leftcord.second)
                           .densevec::outer(other_indexed.get_valat(
                               entry.first, rightcord.second));
              } else {
                auto lop = this_indexed.get_valat(entry.first, leftcord.second);
                auto rop =
                    other_indexed.get_valat(entry.first, rightcord.second);
                outp = lop * rop;
              }
              output[output_coords] = outp;
            }
          }
        }
      }
    }
    // std::cout<<"Output hashmap formed"<<std::endl;
    Tensor<RES> output_tensor(output.size());
    for (auto &entry : output) {
      output_tensor.get_nonzeros().emplace_back(entry.second, entry.first);
    }

    return output_tensor;
  }

  template <class L, class R>
  void fill_values(Tensor<L> &left, Tensor<R> &right, CoOrdinate left_contr,
                   CoOrdinate left_batch, CoOrdinate right_contr,
                   CoOrdinate right_batch) {
    Tensor<DT> result = left.Tensor<L>::template multiply<DT, R>(
        right, left_contr, left_batch, right_contr, right_batch);
    if (this->get_nonzeros().size() != 0) {
      delete_old_values();
    }
    for (auto &nnz : result) {
      auto coords = nnz.get_coords();
      this->get_nonzeros().emplace_back(nnz.get_data(), coords);
    }
    result.get_nonzeros().clear();
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
#define OVERLOAD_OP(OP)                                                        \
  void operator OP(Tensor<DT> &other) {                                        \
    hashmap_vals indexed_tensor;                                               \
    for (auto &nnz : nonzeros) {                                               \
      indexed_tensor[nnz.get_coords()] = nnz.get_data();                       \
    }                                                                          \
    nonzeros.clear();                                                          \
    for (auto &nnz : other) {                                                  \
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
  OVERLOAD_OP(+=)
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
