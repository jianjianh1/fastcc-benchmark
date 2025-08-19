#include <ankerl/unordered_dense.h>
#include <tsl/hopscotch_map.h>
#include <tsl/hopscotch_set.h>
#include <emhash/hash_table8.hpp>
#include "coordinate.hpp"
#include "timer.hpp"
#include <omp.h>
#include <forward_list>
#include <list>
#include <random>
#include <cmath>

template <class DT> class CompactNNZ {
  CompactCordinate cord;
  DT data;

public:
  CompactNNZ(DT data, CompactCordinate cord) : data(data), cord(cord) {}
  DT get_data() { return data; }
  CompactCordinate get_cord() { return cord; }
};

template <class DT> class AtomicNodeNNZ {
  CompactCordinate cord;
  DT data;

public:
  std::atomic<AtomicNodeNNZ<DT> *> next;
  AtomicNodeNNZ(DT data, CompactCordinate cord)
      : data(data), cord(cord), next(nullptr) {}
  DT get_data() { return data; }
  CompactCordinate get_cord() { return cord; }
};

template <class DT> class Tensor;

template <class DT> class AtomicListTensor {
  std::atomic<AtomicNodeNNZ<DT> *> head = nullptr;
  int dimensionality = 42;

public:
  AtomicListTensor(int dimensionality = 0) {
    this->dimensionality = dimensionality;
  }

  AtomicListTensor(AtomicListTensor<DT> &&other) {
    head.store(other.head.load());
    // other.head = nullptr;
    dimensionality = other.dimensionality;
  }

  void push_nnz(DT data, CompactCordinate cord) {
    AtomicNodeNNZ<DT> *new_node = new AtomicNodeNNZ<DT>(data, cord);
    AtomicNodeNNZ<DT> *old_head = head;
    do {
      old_head = head.load();
      new_node->next = old_head;
    } while (!head.compare_exchange_weak(old_head, new_node));
  }
  int get_nnz_count() {
    int count = 0;
    for (AtomicNodeNNZ<DT> *current = head; current != nullptr;
         current = current->next.load()) {
      count++;
    }
    return count;
  }
};
template <class DT> class NNZNode {
private:
  CompactNNZ<DT> nnz;
  NNZNode<DT> *next = nullptr;

public:
  NNZNode(DT data, CompactCordinate cord) : nnz(data, cord), next(nullptr) {}
  NNZNode<DT> *get_next() { return next; }
  void set_next(NNZNode<DT> *next) { this->next = next; }
};

template <class DT> class ListTensor {
  NNZNode<DT>* head = nullptr;
  NNZNode<DT>* tail = nullptr;
  int dimensionality = 0;
  int thread_id = 0;
  uint64_t count = 0;

public:
  ListTensor(int dimensionality = 0, int thread_id=0):dimensionality(dimensionality), thread_id(thread_id) {}

  void push_nnz(DT data, CompactCordinate cord) {
    NNZNode<DT>* new_node = (NNZNode<DT>*)my_malloc(sizeof(NNZNode<DT>), thread_id);
    *new_node = NNZNode<DT>(data, cord);
    this->count++;
    if(head == tail && head == nullptr){
        head = new_node;
        tail = new_node;
    } else if(head == tail){
        head->set_next(new_node);
        tail = new_node;
    } else {
        tail->set_next(new_node);
        tail = new_node;
    }
  }
  int compute_nnz_count(){
      return count;
  }
  int run_through_nnz(){
      int count = 0;
      for(NNZNode<DT>* current = head; current != nullptr; current = current->get_next()){
          count++;
      }
      return count;
  }
  void concatenate(ListTensor& other){
      if(this->tail == nullptr){
          this->head = other.head;
          this->tail = other.tail;
          this->count += other.count;
          return;
      }
      this->tail->set_next(other.head);
      this->tail = other.tail;
      count += other.count;
  }
};

template <class DT> class CompactTensor {
  uint64_t num_nonzeros = 0;
  CompactNNZ<DT> *nonzeros = nullptr;
  // int iter = 0;
  uint64_t iter = 0;
  int dimensionality = 42;

public:
  CompactTensor(uint64_t num_nonzeros, int dimensionality = 0) {
    this->num_nonzeros = num_nonzeros;
    nonzeros = (CompactNNZ<DT> *)malloc(num_nonzeros * sizeof(CompactNNZ<DT>));
    this->dimensionality = dimensionality;
  }

  CompactTensor<DT> cut_at(int position, int extent) {
    assert(position < num_nonzeros);
    assert(extent < num_nonzeros);
    assert(position + extent <= num_nonzeros);
    CompactTensor<DT> result(extent, this->dimensionality);
    result.nonzeros = nonzeros + position;
    return result;
  }
  void push_nnz(DT data, CompactCordinate cord) {
    if (iter >= num_nonzeros) {
      std::cerr << "Tried to push more nonzeros than allocated" << std::endl;
      std::cerr << "Current iter is " << iter << std::endl;
      exit(1);
    }
    nonzeros[iter++] = CompactNNZ<DT>(data, cord);
  }
  int get_reserved_count() { return num_nonzeros; }
  int get_nnz_count() { return iter; }
  Tensor<DT> to_tensor();
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

template <class DT> class SmallIndexedTensor {
  // using hashmap_vals =
  //     emhash8::HashMap<uint64_t, std::vector<std::pair<uint64_t, DT>>>;
  using hashmap_vals =
      ankerl::unordered_dense::map<uint64_t,
                                   std::vector<std::pair<uint64_t, DT>>>;
  // using hashmap_vals =
  //     tsl::hopscotch_map<uint64_t, std::vector<std::pair<uint64_t, DT>>>;

public:
  hashmap_vals indexed_tensor;
  int *shape = nullptr;
  uint64_t max_inner_val = 0;

  using iterator = typename hashmap_vals::iterator;
  using value_type = typename hashmap_vals::value_type;
  iterator begin() { return indexed_tensor.begin(); }
  iterator end() { return indexed_tensor.end(); }

  SmallIndexedTensor(Tensor<DT> &base_tensor, CoOrdinate index_coords) {
    //base_tensor._infer_shape();
    auto removed_shape = base_tensor.get_nonzeros()[0].get_coords().remove(index_coords).get_shape();
    for (auto &nnz : base_tensor) {
      uint64_t index = nnz.get_coords().gather_linearize(index_coords);
      uint64_t remaining = nnz.get_coords().remove_linearize(index_coords, removed_shape);
      if (remaining >= max_inner_val) {
        max_inner_val = remaining;
      }
      auto it = indexed_tensor.find(index);
      if (it != indexed_tensor.end()) {
        it->second.push_back({remaining, nnz.get_data()});
        // it.value().push_back({remaining, nnz.get_data()});
      } else {
        indexed_tensor[index] = {{remaining, nnz.get_data()}};
      }
    }
  }

  uint64_t get_size() { return indexed_tensor.size(); }
  uint64_t get_linearization_bound() { return max_inner_val; }
  uint64_t row_size_of(uint64_t rowind) {
    auto it = indexed_tensor.find(rowind);
    if (it == indexed_tensor.end()) {
      return 0;
    }
    return uint64_t(it->second.size());
  }
};

void make_next_power_of_two(std::vector<int> &shape) {
  for (int i = 0; i < shape.size(); i++) {
    if ((shape[i] & shape[i] - 1) != 0) {
      // copied from stack overflow post
      // https://stackoverflow.com/questions/466204/rounding-up-to-next-power-of-2
      shape[i]--;
      shape[i] |= shape[i] >> 1;
      shape[i] |= shape[i] >> 2;
      shape[i] |= shape[i] >> 4;
      shape[i] |= shape[i] >> 8;
      shape[i] |= shape[i] >> 16;
      shape[i]++;
    } else{
        shape[i] = shape[i]<<1;
    }
    shape[i] = int(log2(shape[i]));
  }
  return;
}

int make_next_power_of_two(int something) {
  if ((something & (something - 1)) != 0) {
    // copied from stack overflow post
    // https://stackoverflow.com/questions/466204/rounding-up-to-next-power-of-2
    something--;
    something |= something >> 1;
    something |= something >> 2;
    something |= something >> 4;
    something |= something >> 8;
    something |= something >> 16;
    something++;
  }
  return something;
}

template <class DT> class TileIndexedTensor {
  using inner_list = std::vector<std::pair<uint64_t, DT>>;
  using middle_map = ankerl::unordered_dense::map<uint64_t, inner_list>;
  // Not sure about this, could be a vector if tiles are not degenerate.
  using tile_map = middle_map*;

public:
  tile_map indexed_tensor;
  int *shape = nullptr;
  int tile_size = 0;
  uint64_t ntiles = 0;
  uint64_t max_inner_val = 0;

  ~TileIndexedTensor() { delete indexed_tensor; }//make it leak, for now.
  TileIndexedTensor(){}
  TileIndexedTensor(Tensor<DT> &base_tensor, CoOrdinate index_coords,
                    int tile_size)
      : tile_size(tile_size) {
    // Tile the dense space 0 to max_inner_val.
    if (this->tile_size == 0) {
      this->tile_size = 1;
    }
    std::vector<int> removed_shape = base_tensor.get_nonzeros()[0]
                                         .get_coords()
                                         .remove(index_coords)
                                         .get_shape();
    uint64_t span = 1;
    for (auto &cord : removed_shape) {
      span *= (cord + 1);
    }
    if(this->tile_size == -1){
        this->tile_size = make_next_power_of_two(span);
    }
    this->ntiles = (span / this->tile_size) + (span % (this->tile_size) != 0);
    this->indexed_tensor = (middle_map *)calloc(this->ntiles, sizeof(middle_map));
    for(int i = 0; i < this->ntiles; i++){
        indexed_tensor[i] = middle_map();
    }
    int num_threads = std::min((uint64_t)this->ntiles, (uint64_t)std::thread::hardware_concurrency()/2);
#pragma omp parallel num_threads(num_threads) shared(indexed_tensor)
    for (auto &nnz : base_tensor) {
      uint64_t remaining = nnz.get_coords().remove_linearize(index_coords, removed_shape);
      uint64_t tile = remaining / this->tile_size;
      if (tile % num_threads != omp_get_thread_num())
          continue;
      uint64_t contraction_index = nnz.get_coords().gather_linearize(index_coords);
      uint64_t inner = remaining % this->tile_size;
      DT data = nnz.get_data();
      middle_map &middle_slice = indexed_tensor[tile];
      auto inner_slice = middle_slice.find(contraction_index);
      if (inner_slice != middle_slice.end()) {
          inner_slice->second.push_back({inner, data});
      } else {
          middle_slice[contraction_index] = {{inner, data}}; // bottleneck 1
      }
    }
  }

  uint64_t get_size() {
    uint64_t count = 0;
    for (uint64_t i = 0; i < this->ntiles; i++) {
      for (auto &inner : indexed_tensor[i]) {
        count += inner.second.size();
      }
    }
    return count;
  }
  uint64_t get_linear_index(uint64_t tile_index, uint64_t index_in_tile) {
    return tile_index * tile_size + index_in_tile;
  }
  uint64_t num_tiles() { return this->ntiles; }
  uint64_t num_nnz_in_tile(uint64_t tile_index) {
    uint64_t count = 0;
    for (auto &inner : indexed_tensor[tile_index]) {
      count += inner.second.size();
    }
    return count;
  }
  float nnz_per_active_column(uint64_t tile_index) {
      float numerator = float(this->num_nnz_in_tile(tile_index));
      float denominator = float(this->indexed_tensor[tile_index].size()); //number of active columns
      if(denominator == 0){
          assert(numerator == 0);
          return 0.0;
      }
      return numerator/denominator;
  }
  //returns a histogram of contraction index(index cord) and the number of tiles in which that shows up.
  std::unordered_map<uint64_t, uint64_t> idx_freq(){
      std::unordered_map<uint64_t, uint64_t> freq;
      for(uint64_t i = 0; i < this->ntiles; i++){
          for(auto& inner : indexed_tensor[i]){
              if(freq.find(inner.first) == freq.end()){
                  freq[inner.first] = 1;
              } else {
                  freq[inner.first]++;
              }
          }
      }
      return freq;
  }
  uint64_t nnz_in_idx_cord(uint64_t c_pos){
      // Count the total number of nonzeros in the indexed tensor at a given position in C (column).
      // Sum over all tiles.
      uint64_t count = 0;
      for(uint64_t i = 0; i < this->ntiles; i++){
          auto it = this->indexed_tensor[i].find(c_pos);
          if(it != this->indexed_tensor[i].end()){
              count += it->second.size();
          }
      }
      return count;
  }
  uint64_t num_active_columns(uint64_t tile_index) {
      return this->indexed_tensor[tile_index].size();
  }
};

template <class DT> class IndexedTensor {
  // using hashmap_vals =
  //     tsl::hopscotch_map<BoundedCoordinate,
  //                        std::vector<std::pair<BoundedCoordinate, DT>>>;
  using hashmap_vals =
      emhash8::HashMap<BoundedCoordinate,
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
    int max_indexed_shape = 0;
    for (auto &icord : index_coords) {
      max_indexed_shape =
          shape[icord] > max_indexed_shape ? shape[icord] : max_indexed_shape;
    }
    indexed_tensor.reserve(max_indexed_shape);
    for (auto &nnz : base_tensor) {
      BoundedCoordinate full_coordinate = nnz.get_coords().get_bounded(shape);
      BoundedCoordinate index = full_coordinate.gather(filter_pos);
      BoundedCoordinate remaining = full_coordinate.remove(filter_pos);
      auto it = indexed_tensor.find(index);
      if (it != indexed_tensor.end()) {
        // it.value().push_back({remaining, nnz.get_data()});
        it->second.push_back({remaining, nnz.get_data()});
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
  uint64_t row_size_of(BoundedCoordinate& rowind) {
    auto it = indexed_tensor.find(rowind);
    if (it == indexed_tensor.end()) {
      return 0;
    }
    return uint64_t(it->second.size());
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

template <class DT>
DT sort_join(std::vector<std::pair<uint64_t, DT>> &left,
             std::vector<std::pair<uint64_t, DT>> &right) {
  std::vector<std::pair<uint64_t, DT>> &smallref =
      left.size() < right.size() ? left : right;
  std::vector<std::pair<uint64_t, DT>> &largeref =
      left.size() < right.size() ? right : left;
  std::sort(smallref.begin(), smallref.end(),
            [](auto a, auto b) { return a.first < b.first; });
  // linear on the smaller vector, binary search on the larger.
  DT sum = DT();
  for (auto p1 = largeref.begin(); p1 != largeref.end(); p1++) {
    auto it = std::lower_bound(smallref.begin(), smallref.end(), p1->first,
                               [](auto a, auto b) { return a.first < b; });
    if (it != smallref.end() && it->first == p1->first) {
      sum += it->second * p1->second;
    }
  }
  return sum;
}
template <class DT>
DT hash_join(std::vector<std::pair<uint64_t, DT>> &left,
             std::vector<std::pair<uint64_t, DT>> &right) {
  using hashmap = ankerl::unordered_dense::map<uint64_t, DT>;
  std::vector<std::pair<uint64_t, DT>> &smallref =
      left.size() < right.size() ? left : right;
  std::vector<std::pair<uint64_t, DT>> &largeref =
      left.size() < right.size() ? right : left;
  hashmap map;
  map.reserve(smallref.size());
  for (auto &p : smallref) {
    map[p.first] = p.second;
  }
  DT sum = DT();
  for (auto &p : largeref) {
    auto it = map.find(p.first);
    if (it != map.end()) {
      sum += it->second * p.second;
    }
  }
  return sum;
}

template <class DT> class OutputTensorHashMap2D {
private:
  using lowest_map = emhash8::HashMap<uint64_t, DT>;
  std::forward_list<std::pair<uint64_t, lowest_map>> nonzeros;

public:
  void add_row(uint64_t left_cord) { nonzeros.push_front({left_cord, lowest_map(5000)}); }
  void set_row(uint64_t left_cord) {
    nonzeros = {{left_cord, {}}};
  }
  void update_last_row(uint64_t right_cord, DT data) {
    lowest_map &current_lowest = nonzeros.front().second;
    auto col_entry = current_lowest.find(right_cord);
    if (col_entry != current_lowest.end()) {
      col_entry->second += data;
    } else {
      current_lowest[right_cord] = data;
    }
  }

  CompactTensor<DT> drain(BoundedCoordinate &sample_left,
                          BoundedCoordinate &sample_right) {
    CompactTensor<DT> result(this->get_nnz_count(),
                             sample_left.get_dimensionality() +
                                 sample_right.get_dimensionality());
    this->drain_into(result, sample_left, sample_right);
    return result;
  }
  template<class SomeTensor> void drain_into(SomeTensor &result, BoundedCoordinate &sample_left,
                  BoundedCoordinate &sample_right) {
    for (auto &first_slice : nonzeros) {
      for (auto &nnz : first_slice.second) {
        CompactCordinate this_cord = CompactCordinate(
            first_slice.first, sample_left, nnz.first, sample_right);
        result.push_nnz(nnz.second, this_cord);
      }
    }
  }
  int get_nnz_count() {
    int count = 0;
    for (auto &top_slice : nonzeros) {
      count += top_slice.second.size();
    }
    return count;
  }
};

template <class DT> class TileAccumulatorDense {
private:
  DT *data_accumulator;
  int left_tile_dim = 0;
  int right_tile_dim = 0;
  int left_tile_index = 0;
  int right_tile_index = 0;
  int thread_id;

public:
  TileAccumulatorDense(int left_tile_dim, int right_tile_dim, int thread_id = 0)
      : left_tile_dim(left_tile_dim), right_tile_dim(right_tile_dim), thread_id(thread_id) {
    int tile_area = left_tile_dim * right_tile_dim;
    this->data_accumulator = (DT *)calloc(tile_area, sizeof(DT));
  }
  void reset_accumulator(int left_tile_index, int right_tile_index) {
    this->left_tile_index = left_tile_index;
    this->right_tile_index = right_tile_index;
  }
  void update(uint64_t pos, DT val) {
    this->data_accumulator[pos] += val;
  }
  template <class TensorType, class BCType>
  void drain_into(TensorType &result_tensor, BCType &sample_left,
                  BCType &sample_right) {
    for (int i = 0; i < left_tile_dim; i++) {
      for (int j = 0; j < right_tile_dim; j++) {
        if (data_accumulator[i * right_tile_dim + j] == DT())
          continue;
        uint64_t left_index = this->left_tile_index * left_tile_dim + i;
        uint64_t right_index = this->right_tile_index * right_tile_dim + j;
        CompactCordinate res_cord = CompactCordinate(left_index, sample_left,
                                                     right_index, sample_right, thread_id);
        result_tensor.push_nnz(data_accumulator[i * right_tile_dim + j],
                               res_cord);
        data_accumulator[i * right_tile_dim + j] = DT();
      }
    }
  }
};

template <class DT> class TileAccumulatorMap {
  using accmap = ankerl::unordered_dense::map<uint64_t, DT>;

private:
  accmap accumulator;
  uint64_t left_tile_dim = 0, right_tile_dim = 0;
  uint64_t left_tile_index = 0, right_tile_index = 0;
  int thread_id;

public:
  TileAccumulatorMap(int left_tile_dim, int right_tile_dim, int thread_id = 0)
      : left_tile_dim(left_tile_dim), right_tile_dim(right_tile_dim), thread_id(thread_id) {
    uint64_t tile_area = left_tile_dim * right_tile_dim;
    accumulator = accmap(20000);
  }
  void reset_accumulator(int left_tile_index, int right_tile_index) {
    this->left_tile_index = left_tile_index;
    this->right_tile_index = right_tile_index;
    accumulator.clear();
  }
  void update(uint64_t pos, DT val) {
    auto it = this->accumulator.find(pos);
    if (it != this->accumulator.end())
      it->second += val;
    else
      this->accumulator[pos] = val;
  }
  template <class TensorType, class BCType>
  void drain_into(TensorType &result_tensor, BCType &sample_left,
                  BCType &sample_right) {
    for (auto &p : accumulator) {
      uint64_t i = p.first / right_tile_dim;
      uint64_t j = p.first % right_tile_dim;
      uint64_t left_index = this->left_tile_index * left_tile_dim + i;
      uint64_t right_index = this->right_tile_index * right_tile_dim + j;
      CompactCordinate this_cord =
          CompactCordinate(left_index, sample_left, right_index, sample_right, thread_id);
      result_tensor.push_nnz(p.second, this_cord);
    }
  }
};


template <class DT, typename bitmask_type> class maskedAccumulator {
private:
  DT *data_accumulator;
  bitmask_type *bitmask;
  uint64_t *active_positions;
  uint64_t pos_iter = 0;
  uint64_t global_count = 0;
  int num_tiles = 1;
  uint64_t left_tile_dim = 0;
  uint64_t right_tile_dim = 0;
  int left_tile_index = 0;
  int right_tile_index = 0;
  int thread_id;

  static const uint64_t n_bits = sizeof(bitmask_type)*8;

public:
  maskedAccumulator(uint64_t left_tile_dim, uint64_t right_tile_dim, int thread_id = 0)
      : left_tile_dim(left_tile_dim), right_tile_dim(right_tile_dim), thread_id(thread_id) {
    uint64_t tile_area = left_tile_dim * right_tile_dim;
    assert(((right_tile_dim & (right_tile_dim - 1)) == 0));
    //this->data_accumulator = (DT *)malloc(tile_area * sizeof(DT));
    this->data_accumulator = (DT *)my_calloc(tile_area, sizeof(DT), thread_id);
    //this->bitmask = (uint8_t *)malloc((tile_area / 8) + 1);

    uint64_t n_bitmasks = (tile_area-1)/(sizeof(bitmask_type)*8)+1;

    this->bitmask = (bitmask_type *) my_calloc(n_bitmasks,sizeof(bitmask_type),thread_id);
    //this->bitmask = (uint8_t *)my_calloc((tile_area / 8 + 1), 1, thread_id);
    //this->active_positions = (uint64_t *)malloc(tile_area * sizeof(uint64_t));
    this->active_positions = (uint64_t *)my_calloc(n_bitmasks, sizeof(uint64_t), thread_id);
  }
  void reset_accumulator(int left_tile_index, int right_tile_index) {
    this->left_tile_index = left_tile_index;
    this->right_tile_index = right_tile_index;
    //int tile_area = left_tile_dim * right_tile_dim;
    //std::fill(this->data_accumulator, this->data_accumulator + tile_area, DT());
    //std::fill(this->bitmask, this->bitmask + (tile_area / 8) + 1, 0);
    //std::fill(this->active_positions, this->active_positions + tile_area, 0);
  }
  void update(uint64_t pos, DT val) {

    bitmask_type bitpos = 1ULL << (pos % n_bits);
    uint64_t bytepos = pos / n_bits;

    bitmask_type read = this->bitmask[bytepos];
    if (read == 0ULL){
      active_positions[this->pos_iter++] = bytepos;
    }
    
    if ((read & bitpos) == 0) {
      
      this->bitmask[bytepos] |= bitpos;
    }

    this->data_accumulator[pos] += val;
  }


  template <class TensorType, class BCType>
  void drain_into(TensorType &result_tensor, BCType& sample_left,
             BCType& sample_right) {
    for (int iter = 0; iter < this->pos_iter; iter++) {

      uint64_t active_bitmask_index = active_positions[iter];

      //where do these start?
      uint64_t starting_dim = active_bitmask_index*n_bits;
      bitmask_type bits = bitmask[active_bitmask_index];

      while (bits != 0ULL){

        int leader = __builtin_ffsll(bits)-1;

        uint64_t true_index = starting_dim+leader;

        uint64_t i = true_index >> uint64_t(log(right_tile_dim));
        uint64_t j = true_index & (right_tile_dim - 1);

        uint64_t left_index = this->left_tile_index * left_tile_dim + i;
        uint64_t right_index = this->right_tile_index * right_tile_dim + j;
        CompactCordinate this_cord = CompactCordinate(left_index, sample_left, right_index, sample_right, thread_id);
        //this_cord.concat(sample_right.delinearize(right_index));
        result_tensor.push_nnz(data_accumulator[true_index],
                             this_cord);
        //wipe after write.
        data_accumulator[true_index] = DT();
        //XOR out leader
        bits ^= 1ULL << leader;


      }

      //unset data
      bitmask[active_bitmask_index] = 0;
      active_positions[iter] = 0;

    }
    this->global_count += pos_iter;
    this->pos_iter = 0;
    this->num_tiles++;
  }
  float percentage_saving(){
      return 1.0 - float(this->global_count)/float(this->num_tiles * this->left_tile_dim * this->right_tile_dim);

  }
};

template <class DT> class TileAccumulator {
private:
  DT *data_accumulator;
  uint8_t *bitmask;
  uint64_t *active_positions;
  uint64_t pos_iter = 0;
  uint64_t global_count = 0;
  int num_tiles = 1;
  uint64_t left_tile_dim = 0;
  uint64_t right_tile_dim = 0;
  int left_tile_index = 0;
  int right_tile_index = 0;
  int thread_id;

public:
  TileAccumulator(uint64_t left_tile_dim, uint64_t right_tile_dim, int thread_id = 0)
      : left_tile_dim(left_tile_dim), right_tile_dim(right_tile_dim), thread_id(thread_id) {
    uint64_t tile_area = left_tile_dim * right_tile_dim;
    assert(((right_tile_dim & (right_tile_dim - 1)) == 0));
    //this->data_accumulator = (DT *)malloc(tile_area * sizeof(DT));
    this->data_accumulator = (DT *)my_calloc(tile_area, sizeof(DT), thread_id);
    //this->bitmask = (uint8_t *)malloc((tile_area / 8) + 1);
    this->bitmask = (uint8_t *)my_calloc((tile_area / 8 + 1), 1, thread_id);
    //this->active_positions = (uint64_t *)malloc(tile_area * sizeof(uint64_t));
    this->active_positions = (uint64_t *)my_calloc(tile_area, sizeof(uint64_t), thread_id);
  }
  void reset_accumulator(int left_tile_index, int right_tile_index) {
    this->left_tile_index = left_tile_index;
    this->right_tile_index = right_tile_index;
    //int tile_area = left_tile_dim * right_tile_dim;
    //std::fill(this->data_accumulator, this->data_accumulator + tile_area, DT());
    //std::fill(this->bitmask, this->bitmask + (tile_area / 8) + 1, 0);
    //std::fill(this->active_positions, this->active_positions + tile_area, 0);
  }
  void update(uint64_t pos, DT val) {
    uint8_t bitpos = 1 << (7 - (pos % 8));
    uint64_t bytepos = pos / 8;
    uint8_t old_mask = this->bitmask[bytepos] & bitpos;
    if (old_mask == 0) {
      active_positions[this->pos_iter++] = pos;
      this->bitmask[bytepos] += bitpos;
    }
    this->data_accumulator[pos] += val;
  }
  template <class TensorType, class BCType>
  void drain_into(TensorType &result_tensor, BCType& sample_left,
             BCType& sample_right) {
    for (int iter = 0; iter < this->pos_iter; iter++) {
      uint64_t i = active_positions[iter] >> uint64_t(log(right_tile_dim));
      uint64_t j = active_positions[iter] & (right_tile_dim - 1);
      uint64_t left_index = this->left_tile_index * left_tile_dim + i;
          //left_indexed.get_linear_index(this->left_tile_index, i);
      uint64_t right_index = this->right_tile_index * right_tile_dim + j;
          //right_indexed.get_linear_index(this->right_tile_index, j);
      
      CompactCordinate this_cord = CompactCordinate(left_index, sample_left, right_index, sample_right, thread_id);
      //this_cord.concat(sample_right.delinearize(right_index));
      result_tensor.push_nnz(data_accumulator[active_positions[iter]],
                             this_cord);
      data_accumulator[active_positions[iter]] = DT();
      bitmask[active_positions[iter]/8] = 0;
      active_positions[iter] = 0;
    }
    this->global_count += pos_iter;
    this->pos_iter = 0;
    this->num_tiles++;
  }
  float percentage_saving(){
      return 1.0 - float(this->global_count)/float(this->num_tiles * this->left_tile_dim * this->right_tile_dim);

  }
};

template <class DT> class OutputTensorHashMap3D {
private:
  // using lowest_map =
  //     tsl::hopscotch_map<uint64_t, DT>;
  // using lowest_map = ankerl::unordered_dense::map<uint64_t, DT>;
  using lowest_map = emhash8::HashMap<uint64_t, DT>;
  // using lowest_map = absl::flat_hash_map<uint64_t, DT>;
  using middle_map = tsl::hopscotch_map<uint64_t, lowest_map>;
  std::forward_list<std::pair<uint64_t, middle_map>> nonzeros;
  lowest_map *current_lowest = nullptr;

public:
  bool is_same_row(BoundedCoordinate &left_ext) {
    uint64_t key = left_ext.as_bigint();

    if (this->nonzeros.empty())
      return true;
    if (this->nonzeros.front().first == key)
      return true;
    return false;
  }
  void add_row(const BoundedCoordinate &left_ext) {
    uint64_t key = left_ext.as_bigint();
    nonzeros.push_front({key, {}});
  }
  void move_sliceptr(const BoundedCoordinate &left_external_cord,
                     const BoundedCoordinate &batch_cord,
                     size_t size_hint = 0) {
    // assumes you're talking about current row. it won't deduplicate across
    // rows
    uint64_t left_external = left_external_cord.as_bigint();
    uint64_t batch = batch_cord.as_bigint();
    assert(!this->nonzeros.empty());
    assert(this->nonzeros.front().first ==
           left_external); // TODO: can remove before flight.
    middle_map &middle_slice = nonzeros.front().second;
    auto lowest_iter = middle_slice.find(batch);
    if (lowest_iter == middle_slice.end()) {
      // middle_slice[batch] = {};
      // middle_slice[batch] = lowest_map(size_hint* 2);
    }
    current_lowest = &middle_slice[batch];
    // if(size_hint > 10000) current_lowest->reserve(10000);
    // current_lowest->reserve(size_hint);
  }
  void update_last_row(const BoundedCoordinate &right_cord, DT data) {
    // assumes we're in the correct first and middle slice, else no
    // deduplication.
    uint64_t right = right_cord.as_bigint();
    auto col_entry = current_lowest->find(right);
    if (col_entry != current_lowest->end()) {
      col_entry->second += data;
      // col_entry.value() += data;
    } else {
      (*current_lowest)[right] = data;
    }
  }

  CompactTensor<DT> drain(BoundedCoordinate &sample_batch,
                          BoundedCoordinate &sample_left,
                          BoundedCoordinate &sample_right) {
    CompactTensor<DT> result(this->get_nnz_count(),
                             sample_batch.get_dimensionality() +
                                 sample_left.get_dimensionality() +
                                 sample_right.get_dimensionality());
    this->drain_into(result, sample_batch, sample_left, sample_right);

    return result;
  }
  template<class SomeTensor> void drain_into(SomeTensor &result, BoundedCoordinate &sample_batch,
                  BoundedCoordinate &sample_left,
                  BoundedCoordinate &sample_right) {
    for (auto &first_slice : nonzeros) {
      // CoOrdinate leftex = BoundedCoordinate(first_slice.first,
      // sample_left).as_coordinate();
      for (auto &second_slice : first_slice.second) {
        // CoOrdinate batch = BoundedCoordinate(second_slice.first,
        // sample_batch).as_coordinate();
        for (auto &nnz : second_slice.second) {
          // CoOrdinate rightex = BoundedCoordinate(nnz.first,
          // sample_right).as_coordinate();
          CompactCordinate this_cord = CompactCordinate(
              second_slice.first, sample_batch, first_slice.first, sample_left,
              nnz.first, sample_right);
          result.push_nnz(nnz.second, this_cord);
        }
      }
    }
  }
  std::unordered_map<int, int> get_lowest_level_histogram() {
    std::unordered_map<int, int> result;
    for (auto &first_slice : nonzeros) {
      for (auto &second_slice : first_slice.second) {
        int sizeof_thistable = second_slice.second.size();
        auto maybe_itr = result.find(sizeof_thistable);
        if (maybe_itr == result.end())
          result[sizeof_thistable] = 1;
        else
          result[sizeof_thistable] += 1;
      }
    }
    return result;
  }
  int get_nnz_count() {
    int count = 0;
    for (auto &top_slice : nonzeros) {
      for (auto &middle_slice : top_slice.second) {
        count += middle_slice.second.size();
      }
    }
    return count;
  }
};

template <class DT> class OutputTensorSort {
private:
  using lowest_map =
      std::vector<BigintNNZ<DT>>; // TODO: this is of size 80k at max,
                                  // seems like. maybe do SMJ
  using middle_map = tsl::hopscotch_map<uint64_t, lowest_map>;
  std::forward_list<std::pair<uint64_t, middle_map>> nonzeros;
  lowest_map *current_lowest = nullptr;

public:
  bool is_same_row(BoundedCoordinate &left_ext) {
    uint64_t key = left_ext.as_bigint();

    if (this->nonzeros.empty())
      return true;
    if (this->nonzeros.front().first == key)
      return true;
    return false;
  }
  void add_row(const BoundedCoordinate &left_ext) {
    uint64_t key = left_ext.as_bigint();
    nonzeros.push_front({key, {}});
  }
  void move_sliceptr(const BoundedCoordinate &left_external_cord,
                     const BoundedCoordinate &batch_cord) {
    // assumes you're talking about current row. it won't deduplicate across
    // rows
    uint64_t left_external = left_external_cord.as_bigint();
    uint64_t batch = batch_cord.as_bigint();
    assert(!this->nonzeros.empty());
    assert(this->nonzeros.front().first ==
           left_external); // TODO: can remove before flight.
    middle_map &middle_slice = nonzeros.front().second;
    auto lowest_iter = middle_slice.find(batch);
    if (lowest_iter == middle_slice.end()) {
      middle_slice[batch] = {};
    }
    current_lowest = &middle_slice[batch];
  }
  void update_last_row(const BoundedCoordinate &right_cord, DT data) {
    // assumes we're in the correct first and middle slice, else no
    // deduplication.
    uint64_t right = right_cord.as_bigint();
    current_lowest->emplace_back(right, data);
  }
  CompactTensor<DT> drain(BoundedCoordinate &sample_batch,
                          BoundedCoordinate &sample_left,
                          BoundedCoordinate &sample_right) {
    CompactTensor<DT> result(this->get_nnz_count(),
                             sample_batch.get_dimensionality() +
                                 sample_left.get_dimensionality() +
                                 sample_right.get_dimensionality());
    this->drain_into(result, sample_batch, sample_left, sample_right);

    return result;
  }
  void drain_into(CompactTensor<DT> &result, BoundedCoordinate &sample_batch,
                  BoundedCoordinate &sample_left,
                  BoundedCoordinate &sample_right) {
    for (auto &first_slice : nonzeros) {
      // CoOrdinate leftex = BoundedCoordinate(first_slice.first,
      // sample_left).as_coordinate();
      for (auto s = first_slice.second.begin(); s != first_slice.second.end();
           s++) {
        auto batch_cord = s->first;
        auto &vecref = s.value();
        // CoOrdinate batch = BoundedCoordinate(second_slice.first,
        // sample_batch).as_coordinate();
        std::sort(vecref.begin(), vecref.end(), [](auto a, auto b) {
          return a.get_bigint() < b.get_bigint();
        });
        DT sum = DT();
        uint64_t last = vecref[0].get_bigint();
        for (auto &nnz : vecref) {
          if (nnz.get_bigint() == last) {
            sum += nnz.get_value();
            continue;
          }
          // CoOrdinate rightex = BoundedCoordinate(nnz.first,
          // sample_right).as_coordinate();
          CompactCordinate this_cord =
              CompactCordinate(batch_cord, sample_batch, first_slice.first,
                               sample_left, last, sample_right);
          result.push_nnz(nnz.get_value(), this_cord);
          last = nnz.get_bigint();
          sum = nnz.get_value();
        }
        CompactCordinate this_cord =
            CompactCordinate(batch_cord, sample_batch, first_slice.first,
                             sample_left, last, sample_right);
        result.push_nnz(sum, this_cord);
      }
    }
  }
  size_t get_nnz_count() {
    size_t count = 0;
    for (auto &top_slice : nonzeros) {
      for (auto &middle_slice : top_slice.second) {
        count += middle_slice.second.size();
      }
    }
    return count;
  }
};


