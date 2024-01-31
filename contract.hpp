#include <algorithm>
#include <boost/functional/hash.hpp>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>

template <typename T> static std::size_t hasharray(int size, T *arr) {
  return boost::hash_range(arr, arr + size);
}

class CoOrdinate {
  int *coords;
  int dimensionality;

public:
  CoOrdinate(int dimensionality, int *coords) {
    int *mymem = new int[dimensionality];
    memcpy(mymem, coords, dimensionality * sizeof(int));
    this->dimensionality = dimensionality;
    this->coords = mymem;
  }
  std::pair<int *, int> get_ref() const {
    return std::make_pair(coords, dimensionality);
  }
  int get_index(int dim) { return coords[dim]; }
  bool operator==(const CoOrdinate &other) const {
    // TODO we might have to make all permutations equal to one another.
    if (dimensionality != other.dimensionality) {
      return false;
    }
    for (int i = 0; i < dimensionality; i++) {
      if (coords[i] != other.coords[i]) {
        return false;
      }
    }
    return true;
  }
};

template <> struct std::hash<CoOrdinate> {
  std::size_t operator()(const CoOrdinate &c) const {
    std::string catted_cord = "";
    auto [coords, dim] = c.get_ref();
    for (int i = 0; i < dim; i++) {
      catted_cord += std::to_string(coords[i]);
      catted_cord += ",";
    }
    return std::hash<std::string>{}(catted_cord);
  }
};

class NNZ {
  float data;
  std::vector<int> coords;

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
    data = disf(gen);
    for (int i = 0; i < dimensionality; i++) {
      this->coords.push_back(dis_cords[i](gen));
    }
  }
  int get_index(int dim) { return coords[dim]; }
  float get_data() { return data; }

  CoOrdinate get_coords() { return CoOrdinate(coords.size(), coords.data()); }
  // careful, this will move out the co-ordinates.

  // Constructor for a given value and coordinates
  NNZ(float data, int dimensionality, int *coords) {
    this->data = data;
    for (int i = 0; i < dimensionality; i++) {
      this->coords.push_back(coords[i]);
    }
  }
};

class Tensor {
private:
  std::vector<NNZ> nonzeros;
  int *shape;
  int dimensionality;
  using hashmap = std::unordered_map<CoOrdinate, int>;

public:
  using iterator = typename std::vector<NNZ>::iterator;
  using value_type = typename std::vector<NNZ>::value_type;
  iterator begin() { return nonzeros.begin(); }
  iterator end() { return nonzeros.end(); }
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
  Tensor(int size) { nonzeros.reserve(size); }
  std::vector<NNZ> &get_nonzeros() { return nonzeros; }
  // void infer_shape() {
  //   for (auto &nnz : this) {
  //     for (int i = 0; i < D; i++) {
  //       shape[i] = std::max(shape[i], nnz.coords[i]);
  //     }
  //   }
  // }
  float get_valat(CoOrdinate &&coords) {
    for (auto &nnz : nonzeros) {
      if (nnz.get_coords() == coords) {
        return nnz.get_data();
      }
    }
    return -1;
  }
  void index_contraction(int *contraction, int num_contr,
                         hashmap &indexed_tensor) {
    for (auto &nnz : *this) {
      int contr[num_contr];
      for (int i = 0; i < num_contr; i++) {
        contr[i] = nnz.get_index(contraction[i]);
      }
      auto filtered_coords = CoOrdinate(num_contr, contr);
      auto ref = indexed_tensor.find(filtered_coords);
      if (ref != indexed_tensor.end()) {
        ref->second += 1;
      } else {
        indexed_tensor[filtered_coords] = 1;
      }
    }
    return;
  }

  int count_ops(Tensor &other, int num_contr, int *left_contr,
                int *right_contr) {
    hashmap first_index;
    this->index_contraction(left_contr, num_contr, first_index);
    hashmap second_index;
    other.index_contraction(right_contr, num_contr, second_index);
    int ops = 0;
    for (auto &entry : first_index) {
      auto ref = second_index.find(entry.first);
      if (ref != second_index.end()) {
        ops += entry.second * ref->second;
      }
    }
    return ops;
  }
};
