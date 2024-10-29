#ifndef COORDINATE_HPP
#define COORDINATE_HPP
// Stack based coordinate system

#include <algorithm>
#include <assert.h>
#include <boost/container_hash/hash.hpp>
#include <cstring>
#include <iostream>
#include <variant>
#include <vector>
#define DIMENSIONALITY 6
#define GOOD_PRIME 3145739

class CoOrdinate;

class BoundedPosition {
  short dimensions = 0;
  int positions[DIMENSIONALITY];

public:
  BoundedPosition(BoundedPosition &left, BoundedPosition &right) {
    for (int i = 0; i < left.get_dimensionality(); i++) {
      positions[i] = left.positions[i];
    }
    dimensions = left.get_dimensionality();
    for (int i = 0; i < right.get_dimensionality(); i++) {
      positions[i + dimensions] = right.positions[i];
    }
    dimensions += right.get_dimensionality();
    assert(dimensions <= DIMENSIONALITY);
  }
  template <class It> BoundedPosition(It begin, It end) {
    for (It iter = begin; iter < end; iter++) {
      positions[dimensions++] = *iter;
    }
  }
  BoundedPosition(std::vector<int> positions) {
    for (int i = 0; i < positions.size(); i++) {
      this->positions[i] = positions[i];
    }
    dimensions = positions.size();
  }
  int get_dimensionality() { return dimensions; }
  int get_at(int position) {
    assert(position < dimensions);
    return positions[position];
  }
  bool has_position(int position) {
    return std::find(positions, positions + dimensions, position) !=
           positions + dimensions;
  }
};

// This is only for tensors with co-ordinates up to 16 bits, ie 65k
class BoundedCoordinate {
  uint8_t dimensions = 0;
  uint16_t coords[DIMENSIONALITY];
  uint16_t bounds[DIMENSIONALITY];
  //size_t linearization = -1;

public:
  BoundedCoordinate(){
      // Give an empty coordinate
      BoundedCoordinate(nullptr, nullptr, 0);
  }
  // Pass in a sample cordinate with the same shape as what you expect for linearized.
  // if you change the as_bigint method, this has to change.
  BoundedCoordinate(uint64_t bigint, BoundedCoordinate const &sample_cord) {
      dimensions = sample_cord.get_dimensionality();
      uint64_t linearized =  bigint;
      for (int iter = dimensions - 1; iter >= 0; iter--) {
      bounds[iter] = sample_cord.get_bound(iter);
      coords[iter] = linearized % (bounds[iter] + 1);
      linearized = (linearized - coords[iter]) / (bounds[iter] + 1);
      }
      assert(this->as_bigint() == bigint); // TODO remove before flight
  }
  BoundedCoordinate(int *coords, int *bounds, int dimensions) {
    for (int i = 0; i < dimensions; i++) {
      this->coords[i] = coords[i];
      this->bounds[i] = bounds[i];
    }
    this->dimensions = dimensions;
    //linearization = this->get_linearization();
  }
  BoundedCoordinate(BoundedCoordinate &left, BoundedCoordinate &right) {
    for (int i = 0; i < left.get_dimensionality(); i++) {
      this->coords[i] = left.coords[i];
      this->bounds[i] = left.bounds[i];
    }
    dimensions = left.get_dimensionality();
    for (int i = 0; i < right.get_dimensionality(); i++) {
      this->coords[i + dimensions] = right.coords[i];
      this->bounds[i + dimensions] = right.bounds[i];
    }
    dimensions += right.get_dimensionality();
    assert(dimensions <= DIMENSIONALITY);
    //linearization = this->get_linearization();
  }
  BoundedCoordinate(BoundedCoordinate &left, BoundedCoordinate &mid,
                    BoundedCoordinate &right) {

    for (int i = 0; i < left.get_dimensionality(); i++) {
      this->coords[i] = left.coords[i];
      this->bounds[i] = left.bounds[i];
    }
    dimensions = left.get_dimensionality();
    for (int i = 0; i < mid.get_dimensionality(); i++) {
      this->coords[i + dimensions] = mid.coords[i];
      this->bounds[i + dimensions] = mid.bounds[i];
    }
    dimensions += mid.get_dimensionality();
    for (int i = 0; i < right.get_dimensionality(); i++) {
      this->coords[i + dimensions] = right.coords[i];
      this->bounds[i + dimensions] = right.bounds[i];
    }
    dimensions += right.get_dimensionality();
    assert(dimensions <= DIMENSIONALITY);
    assert(dimensions == left.get_dimensionality() + mid.get_dimensionality() +
                             right.get_dimensionality());
    //linearization = this->get_linearization();
  }
  CoOrdinate as_coordinate() const;
  uint64_t as_bigint() const{
      return (uint64_t)this->get_linearization();
  }
  int get_dimensionality() const { return dimensions; }
  std::string to_string() const {
    std::string str = "";
    for (int i = 0; i < dimensions; i++) {
      str += std::to_string(this->coords[i]) + "/" +
             std::to_string(this->bounds[i]) + " ";
    }
    return str;
  }

  int get_coordinate(int position) const {
    assert(position < dimensions);
    return coords[position];
  }
  int get_bound(int position) const {
    assert(position < dimensions);
    return bounds[position];
  }
  size_t get_linear_bound(int offset = 1) const {
    size_t result = 1;
    for (int i = 0; i < this->get_dimensionality(); i++) {
      result *= (size_t)(this->get_bound(i) + offset);
    }
    return result;
  }
  size_t get_linearization(int offset = 1) const {
    //if (this->linearization != -1)
    //  return this->linearization;
    size_t linearlized_cord = 0;
    for (int i = 0; i < this->get_dimensionality(); i++) {
      linearlized_cord += this->get_coordinate(i);
      if (i != this->get_dimensionality() - 1) {
        linearlized_cord *= (this->get_bound(i + 1) + offset);
      }
    }
    return linearlized_cord;
  }
  BoundedCoordinate gather(BoundedPosition &other) {
    int res_cords[DIMENSIONALITY];
    int res_bounds[DIMENSIONALITY];
    assert(dimensions >= other.get_dimensionality());
    int res_dimensionality = 0;
    for (int i = 0; i < other.get_dimensionality(); i++) {
      res_cords[res_dimensionality] = coords[other.get_at(i)];
      res_bounds[res_dimensionality++] = bounds[other.get_at(i)];
    }
    assert(res_dimensionality <= DIMENSIONALITY);
    assert(res_dimensionality == other.get_dimensionality());
    return BoundedCoordinate(res_cords, res_bounds, other.get_dimensionality());
  }
  BoundedCoordinate remove(BoundedPosition &other) {
    int res_cords[DIMENSIONALITY];
    int res_bounds[DIMENSIONALITY];
    assert(dimensions >= other.get_dimensionality());
    int res_dimensionality = 0;
    for (int i = 0; i < dimensions; i++) {
      if (other.has_position(i)) {
        // if the position is found in the positions to remove, skip it
        continue;
      }
      res_cords[res_dimensionality] = coords[i];
      res_bounds[res_dimensionality++] = bounds[i];
    }
    assert(res_dimensionality <= DIMENSIONALITY);
    assert(res_dimensionality == dimensions - other.get_dimensionality());
    return BoundedCoordinate(res_cords, res_bounds, res_dimensionality);
  }
  bool operator==(const BoundedCoordinate &other) const {
    if (dimensions != other.get_dimensionality()) {
      return false;
    }
    for (int i = 0; i < dimensions; i++) {
      if (coords[i] != other.get_coordinate(i)) {
        return false;
      }
    }
    return true;
  }
};
template <> struct std::hash<BoundedCoordinate> {
  std::size_t operator()(const BoundedCoordinate &c) const {
    return c.get_linearization();
  }
};

static int doubleequals = 0;

class OutputCoordinate {
  BoundedCoordinate batch, left_external, right_external;

public:
  size_t linearization = -1;
  OutputCoordinate(BoundedCoordinate b, BoundedCoordinate l,
                   BoundedCoordinate r)
      : batch(b), left_external(l), right_external(r) {
    // std::cout << "Batch " << b.to_string() << ", left " << l.to_string()
    //           << ", right " << r.to_string() << std::endl;
    // std::cout<<"Linearization is "<<this->get_linearization()<<std::endl;
    // std::cout<<"Linearization: "<<std::endl;
    // std::cout<<"Batch "<<b.get_linearization()<<", left
    // "<<l.get_linearization()<<", right "<<r.get_linearization()<<std::endl;
    // std::cout<<"Bounds: "<<std::endl;
    // std::cout<<"Batch "<<b.get_linear_bound()<<", left
    // "<<l.get_linear_bound()<<", right "<<r.get_linear_bound()<<std::endl;
    this->linearization = this->get_linearization();
  }
  bool operator==(const OutputCoordinate &other) const {
    doubleequals++;
    return this->linearization == other.linearization;
    // return batch == other.batch && left_external == other.left_external &&
    //        right_external == other.right_external;
  }
  int static get_equality_count() { return doubleequals; }
  BoundedCoordinate merge() {
    return BoundedCoordinate(batch, left_external, right_external);
  }
  const BoundedCoordinate &get_batch() const { return batch; }
  const BoundedCoordinate &get_left() const { return left_external; }
  const BoundedCoordinate &get_right() const { return right_external; }
  size_t get_linearization() const {
    if (this->linearization != -1)
      return this->linearization;
    return (batch.get_linearization() * (left_external.get_linear_bound() *
                                         right_external.get_linear_bound()) +
            left_external.get_linearization() *
                right_external.get_linear_bound() +
            right_external.get_linearization());
  }
  uint64_t as_bigint() const{
      return (uint64_t)(this->linearization);
  }
  size_t get_min_hash() const {
    size_t batch_bound = batch.get_linear_bound();
    size_t left_bound = left_external.get_linear_bound();
    size_t right_bound = right_external.get_linear_bound();
    if (batch_bound >= left_bound && batch_bound >= right_bound)
      return std::hash<BoundedCoordinate>()(batch);
    if (left_bound >= batch_bound && left_bound >= right_bound)
      return std::hash<BoundedCoordinate>()(left_external);
    if (right_bound >= left_bound && right_bound >= batch_bound)
      return std::hash<BoundedCoordinate>()(right_external);
    else
      assert(false);
  }
  //CoOrdinate as_coordinate(){
  //    auto single_cord = this->merge();
  //    return single_cord.as_coordinate();
  //}
  // int get_dimensionality() const {
  //     return batch.get_dimensionality() + left_external.get_dimensionality()
  //     + right_external.get_dimensionality();
  // }
};

template <> struct std::hash<OutputCoordinate> {
  std::size_t operator()(const OutputCoordinate &c) const {
    // size_t result = 0;
    // boost::hash_combine(result,
    // std::hash<BoundedCoordinate>()(c.get_batch()));
    // boost::hash_combine(result,
    // std::hash<BoundedCoordinate>()(c.get_left()));
    // boost::hash_combine(result,
    // std::hash<BoundedCoordinate>()(c.get_right()));
    // return result;
    size_t hash_res =
        c.get_batch().get_linearization(0) * c.get_left().get_linear_bound(0) *
            c.get_right().get_linear_bound(0) +
        c.get_left().get_linearization(0) * c.get_right().get_linear_bound(0) +
        c.get_right().get_linearization(0);
    return std::hash<size_t>()(hash_res);
    // return c.get_min_hash();
    //  size_t linearlized_cord = 0;
    //  for (int i = 0; i < c.get_dimensionality(); i++) {
    //    linearlized_cord += c.get_coordinate(i);
    //    if (i != c.get_dimensionality() - 1) {
    //      linearlized_cord *= c.get_bound(i + 1);
    //    }
    //  }
  }
};

class CoOrdinate {
#define BITWIDTH (512)
  std::vector<int> coords;
  std::bitset<BITWIDTH> mybits;
  std::vector<int> max_indices;

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
      mybits <<= (sizeof(int) * 8);
      mybits |= std::bitset<BITWIDTH>(cord);
    }
  }
  void all_positions(int dimensionality) {
    coords.clear();
    // get 0 to D-1, all positions.
    for (int i = 0; i < dimensionality; i++) {
      coords.push_back(i);
    }
  }
  CoOrdinate(std::vector<int> const &data,
             std::vector<int> shape = std::vector<int>()) {
    this->coords = data;
    for (auto &cord : this->coords) {
      mybits <<= (sizeof(int) * 8);
      mybits |= std::bitset<BITWIDTH>(cord);
    }
    this->max_indices = shape;
  }
  std::string to_string() const;
  void write(std::string filename) const;
  BoundedCoordinate get_bounded(int *bounds) const {
    return BoundedCoordinate((int *)coords.data(), bounds, coords.size());
  }

  // This is going to concatenate two coordinates
  CoOrdinate(CoOrdinate const &left, CoOrdinate const &right) {
    // if(left.get_shape().size() == 0 || right.get_shape().size() == 0){
    //     std::cerr<<"Need to set shape before concatenating
    //     coordinates"<<std::endl; assert(false);
    // } else {
    // this->max_indices.reserve(left.get_dimensionality() +
    // right.get_dimensionality());
    if (left.get_shape().size() > 0) {
      this->max_indices.insert(this->max_indices.end(),
                               left.get_shape().begin(),
                               left.get_shape().end());
    }
    if (right.get_shape().size() > 0) {
      this->max_indices.insert(this->max_indices.end(),
                               right.get_shape().begin(),
                               right.get_shape().end());
    }
    //}
    coords.reserve(left.get_dimensionality() + right.get_dimensionality());
    // memcpy(coords.data(), left.coords.data(), left.get_dimensionality() *
    // sizeof(int)); memcpy(coords.data() + left.get_dimensionality(),
    // right.coords.data(), right.get_dimensionality() * sizeof(int));
    coords.insert(coords.end(), left.coords.begin(), left.coords.end());
    coords.insert(coords.end(), right.coords.begin(), right.coords.end());
    for (auto &cord : this->coords) {
      mybits <<=
          (sizeof(int) *
           8); // sizeof is in bytes, so we need to multiply by 8 to get bits
      mybits |= std::bitset<BITWIDTH>(cord);
    }
  }

  CoOrdinate(CoOrdinate const &left, CoOrdinate const &mid,
             CoOrdinate const &right) {
    // if(left.get_shape().size() == 0 || mid.get_shape().size() == 0 ||
    // right.get_shape().size() == 0){
    //     std::cerr<<"Need to set shape before concatenating
    //     coordinates"<<std::endl; assert(false);
    // } else {
    // this->max_indices.reserve(left.get_dimensionality() +
    // mid.get_dimensionality() + right.get_dimensionality());
    if (left.get_shape().size() > 0) {
      this->max_indices.insert(this->max_indices.end(),
                               left.get_shape().begin(),
                               left.get_shape().end());
    }
    if (mid.get_shape().size() > 0) {
      this->max_indices.insert(this->max_indices.end(), mid.get_shape().begin(),
                               mid.get_shape().end());
    }
    if (right.get_shape().size() > 0) {
      this->max_indices.insert(this->max_indices.end(),
                               right.get_shape().begin(),
                               right.get_shape().end());
    }
    //}
    coords.reserve(left.get_dimensionality() + mid.get_dimensionality() +
                   right.get_dimensionality());
    coords.insert(coords.end(), left.coords.begin(), left.coords.end());
    coords.insert(coords.end(), mid.coords.begin(), mid.coords.end());
    coords.insert(coords.end(), right.coords.begin(), right.coords.end());
    for (auto &cord : this->coords) {
      mybits <<=
          (sizeof(int) *
           8); // sizeof is in bytes, so we need to multiply by 8 to get bits
      mybits |= std::bitset<BITWIDTH>(cord);
    }
  }

  CoOrdinate gather(CoOrdinate const &positions) const {
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
    std::vector<int> gathered_shape;
    std::vector<int> og_shape = this->get_shape();
    if (og_shape.size() > 0) {
      for (auto &cord : positions) {
        gathered_shape.push_back(og_shape[cord]);
      }
    }
    for (int i = 0; i < positions.get_dimensionality(); i++) {
      gathered.push_back(coords[positions.get_index(i)]);
    }
    return CoOrdinate(gathered, gathered_shape);
  }

  CoOrdinate remove(CoOrdinate const &positions) {
    std::vector<int> removed;
    std::vector<int> removed_shape;
    for (int i = 0; i < this->get_dimensionality(); i++) {
      if (std::find(positions.begin(), positions.end(), i) == positions.end()) {
        removed.push_back(coords[i]);
      }
    }
    if (max_indices.size() > 0) {
      for (int i = 0; i < this->get_dimensionality(); i++) {
        if (std::find(positions.begin(), positions.end(), i) ==
            positions.end()) {
          removed_shape.push_back(max_indices[i]);
        }
      }
    }
    return CoOrdinate(removed, removed_shape);
  }

  int get_index(int dim) const { return coords[dim]; }
  int get_dimensionality() const { return coords.size(); }
  void set_shape(std::vector<int> shape) { max_indices = shape; }
  const std::vector<int> &get_shape() const { return max_indices; }
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

CoOrdinate BoundedCoordinate::as_coordinate() const {
  std::vector<int> result;
  for (int i = 0; i < dimensions; i++) {
    result.push_back(coords[i]);
  }
  return result;
}

template <> struct std::hash<CoOrdinate> {
  std::size_t operator()(const CoOrdinate &c) const {
    if (c.get_shape().size() == 0) {
      std::cerr << "Need to set shape before hashing coordinate" << std::endl;
      assert(false);
    }

    size_t linearlized_cord = 0;
    for (int i = 0; i < c.get_dimensionality(); i++) {
      linearlized_cord += c.get_index(i);
      if (i != c.get_dimensionality() - 1) {
        linearlized_cord *= c.get_shape()[i + 1];
      }
    }
    return linearlized_cord;

    // return std::hash<std::bitset<BITWIDTH>>{}(c.get_bits());
  }
};

#endif
