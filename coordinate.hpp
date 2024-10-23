#ifndef COORDINATE_HPP
#define COORDINATE_HPP
// Stack based coordinate system

#include <algorithm>
#include <assert.h>
#include <boost/container_hash/hash.hpp>
#include <cstring>
#include <variant>
#include <vector>
#define DIMENSIONALITY 6
#define GOOD_PRIME 3145739

class BoundedPosition {
  int dimensions = 0;
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

class BoundedCoordinate {
  int dimensions = 0;
  int coords[DIMENSIONALITY];
  int bounds[DIMENSIONALITY];

public:
  BoundedCoordinate(int *coords, int *bounds, int dimensions) {
    for (int i = 0; i < dimensions; i++) {
      this->coords[i] = coords[i];
      this->bounds[i] = bounds[i];
    }
    this->dimensions = dimensions;
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
  }
  int get_dimensionality() const { return dimensions; }

  int get_coordinate(int position) const {
    assert(position < dimensions);
    return coords[position];
  }
  int get_bound(int position) const {
    assert(position < dimensions);
    return bounds[position];
  }
  size_t get_linear_bound() const {
    size_t result = 1;
    for (int i = 0; i < this->get_dimensionality(); i++) {
      result *= this->get_bound(i);
    }
    return result;
  }
  size_t get_linearization() const {
    size_t linearlized_cord = 0;
    for (int i = 0; i < this->get_dimensionality(); i++) {
      linearlized_cord += this->get_coordinate(i);
      if (i != this->get_dimensionality() - 1) {
        linearlized_cord *= this->get_bound(i + 1);
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

class OutputCoordinate {
  BoundedCoordinate batch, left_external, right_external;

public:
  OutputCoordinate(BoundedCoordinate b, BoundedCoordinate l,
                   BoundedCoordinate r)
      : batch(b), left_external(l), right_external(r) {}
  bool operator==(const OutputCoordinate &other) const {
    return batch == other.batch && left_external == other.left_external &&
           right_external == other.right_external;
  }
  BoundedCoordinate merge() {
    return BoundedCoordinate(batch, left_external, right_external);
  }
  const BoundedCoordinate &get_batch() const { return batch; }
  const BoundedCoordinate &get_left() const { return left_external; }
  const BoundedCoordinate &get_right() const { return right_external; }
  size_t get_linearization() const {
    return (batch.get_linearization() * (left_external.get_linear_bound() *
                                         right_external.get_linear_bound()) +
            left_external.get_linearization() *
                right_external.get_linear_bound() +
            right_external.get_linearization());
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
  // int get_dimensionality() const {
  //     return batch.get_dimensionality() + left_external.get_dimensionality()
  //     + right_external.get_dimensionality();
  // }
};

template <> struct std::hash<OutputCoordinate> {
  std::size_t operator()(const OutputCoordinate &c) const {
    //size_t result = 0;
    //boost::hash_combine(result,
    //std::hash<BoundedCoordinate>()(c.get_batch()));
    //boost::hash_combine(result,
    //std::hash<BoundedCoordinate>()(c.get_left()));
    //boost::hash_combine(result,
    //std::hash<BoundedCoordinate>()(c.get_right()));
    //return result;
    return c.get_linearization();
    //return c.get_min_hash();
    // size_t linearlized_cord = 0;
    // for (int i = 0; i < c.get_dimensionality(); i++) {
    //   linearlized_cord += c.get_coordinate(i);
    //   if (i != c.get_dimensionality() - 1) {
    //     linearlized_cord *= c.get_bound(i + 1);
    //   }
    // }
  }
};

#endif
