#ifndef TYPES_H
#define TYPES_H
#include <assert.h>
#include <string.h>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <vector>

class densemat;
class densevec {
  int size = 0;
  double *values=nullptr;

public:
  void clear() { memset(values, 0, sizeof(double) * size); }
  densevec() { }
  densevec(std::vector<double> some_data) {
    size = some_data.size();
    values = new double[size];
    for (int i = 0; i < size; i++) {
      values[i] = some_data[i];
    }
  }
  double operator()(int i) {
    assert(i < size);
    return values[i];
  }
  double *operator&() { return &values[0]; }
  int getsize() { return size; }
  // Now we need to implement the following
  // vec * scalar
  // scalar * vec
  // vec * vec -> inner dot, we don't ever need  element-wise
  densevec operator*(double scalar) const {
    std::vector<double> res_data;
    for (int i = 0; i < size; i++) {
      res_data.push_back(values[i] * scalar);
    }
    densevec result = densevec(res_data);
    return result;
  }
  std::string to_string() const {
    std::string result = "";
    for (int i = 0; i < size; i++) {
      result += std::to_string(values[i]) + " ";
    }
    return result;
  }
  densevec operator+=(densevec other) {
    if (size != other.size) {
      std::cerr << "Vector sizes do not match for an addition. Left is " << size
                << ", while right is " << other.getsize() << std::endl;
      exit(1);
    }
    for (int i = 0; i < size; i++) {
      values[i] += other(i);
    }
    return *this;
  }
  double operator*(densevec other) const {
    if (size != other.size) {
      std::cerr << "Vector sizes do not match for an inner product"
                << std::endl;
      std::cerr << "Left is " << size << ", while right is " << other.getsize()
                << std::endl;
      exit(1);
    }
    double result = 0.0;
    for (int i = 0; i < size; i++) {
      result += values[i] * other(i);
    }
    return result;
  }

  densemat outer(densevec other) const;

  void free() { delete[] values; }
  bool operator==(densevec other) {
    if (size != other.size) {
      return false;
    }
    return (memcmp(values, other.values, sizeof(double) * size) == 0);
  }
};
densevec operator*(double k, densevec other) { return other * k; }

class densemat {
  int size = 0;
  double *values = nullptr;

public:
  void clear() { memset(values, 0, sizeof(double) * size * size); }
  densemat() { }
  void free() {
    if (values != nullptr) {
      delete[] values;
      values = nullptr;
    }
  }
  double *operator&() { return &values[0]; }
  int getsize() { return size; }
  double operator()(int i, int j) {
    assert(i < size && j < size);
    return values[i * size + j];
  }
  densemat(std::vector<double> some_data) {
    int root = int(sqrt(some_data.size()));
    if (root * root != some_data.size()) {
      std::cerr << "Matrix data is not square, size of vec is "
                << some_data.size() << std::endl;
      exit(1);
    }
    size = root;
    values = new double[size * size];
    for (int i = 0; i < size * size; i++) {
      values[i] = some_data[i];
    }
  }
  std::string to_string() {
    std::string result = "";
    for (int i = 0; i < size * size; i++) {
      result += std::to_string(values[i]) + " ";
    }
    return result;
  }
  bool operator==(densemat other) {
    if (size != other.size) {
      return false;
    }
    return (memcmp(values, other.values, sizeof(double) * size * size) ==
            0);
  }
// In-place eltwise operations
#define OVERLOAD_OP(OP)                                                        \
  densemat operator OP(densemat other) {                                       \
    if (size != other.size) {                                                  \
      std::cerr << "Matrix sizes do not match for an eltwise op. Left is "     \
                << size << ", while right is " << other.getsize()              \
                << std::endl;                                                  \
      exit(1);                                                                 \
    }                                                                          \
    for (int i = 0; i < size * size; i++) {                                    \
      values[i] OP other(i / size, i % size);                                  \
    }                                                                          \
    return *this;                                                              \
  }
  OVERLOAD_OP(+=)
  OVERLOAD_OP(-=)
  OVERLOAD_OP(/=)
  //  Following operators for matrices
  //  mat * scalar -> scalar. eltwise
  //  scalar * mat -> scalar. eltwise
  //  mat * mat -> mat. gemm
  //  mat * vec -> vec. gemv
  //  vec * mat -> vec. gemv
  //
  densemat operator*(double scalar) const {
    std::vector<double> res_data;
    for (int i = 0; i < size * size; i++) {
      res_data.push_back(scalar * values[i]);
    }
    densemat result = densemat(res_data);
    return result;
  }
  densemat operator*(densemat other) const {
    if (size != other.size) {
      std::cerr << "Matrix sizes do not match for a matrix product"
                << std::endl;
      std::cerr << "Left is " << size << ", while right is " << other.getsize()
                << std::endl;
      exit(1);
    }
    std::vector<double> res_data;
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        double sum = 0.0;
        for (int k = 0; k < size; k++) {
          sum += values[i * size + k] * other(k, j);
        }
        res_data.push_back(sum);
      }
    }
    densemat result = densemat(res_data);
    return result;
  }
  // Elementwise product followed by sum
  double mult_reduce(densemat other) const {
    if (size != other.size) {
      std::cerr << "Matrix sizes do not match for an inner product"
                << std::endl;
      std::cerr << "Left is " << size << ", while right is " << other.getsize()
                << std::endl;
      exit(1);
    }
    double result = 0.0;
    for (int i = 0; i < size * size; i++) {
      result += values[i] * other(i / size, i % size);
    }
    return result;
  }
  // Row of matrix shows up in result, column is contracted with another vector
  densevec operator*(densevec other) const {
    if (size != other.getsize()) {
      std::cerr
          << "Matrix and vector sizes do not match for a matrix-vector product"
          << std::endl;
      exit(1);
    }
    std::vector<double> res_data;
    for (int i = 0; i < size; i++) {
      double sum = 0.0;
      for (int j = 0; j < size; j++) {
        sum += values[i * size + j] * other(j);
      }
      res_data.push_back(sum);
    }
    densevec result = densevec(res_data);
    return result;
  }
};
densemat densevec::outer(densevec other) const {
  std::vector<double> res_data;
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < other.size; j++) {
      res_data.push_back(values[i] * other(j));
    }
  }
  densemat result = densemat(res_data);
  return result;
}

double operator+=(double prev, densevec vec) {
  for (int i = 0; i < vec.getsize(); i++) {
    prev += vec(i);
  }
  return prev;
}
double operator+=(double prev, densemat mat) {
  for (int i = 0; i < mat.getsize(); i++) {
    prev += mat(i / mat.getsize(), i % mat.getsize());
  }
  return prev;
}

densemat operator*(double k, densemat other) { return other * k; }
densevec operator*(densevec vec, densemat mat) { return mat * vec; }

#endif
