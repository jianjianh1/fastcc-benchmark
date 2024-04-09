#ifndef TYPES_H
#define TYPES_H
#include <cassert>
#include <cstring>
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <vector>

class densemat;
class densevec {
  int size = 0;
  double *values;

public:
  void clear() { std::memset(values, 0, sizeof(double) * size); }
  densevec() { values = new double[size]; }
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
  densevec operator*(double scalar) {
    std::vector<double> res_data;
    for (int i = 0; i < size; i++) {
      res_data.push_back(values[i] * scalar);
    }
    densevec result = densevec(res_data);
    return result;
  }
  std::string to_string() {
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
      return densevec();
    }
    for (int i = 0; i < size; i++) {
      values[i] += other(i);
    }
    return *this;
  }
  double operator*(densevec other) {
    if (size != other.size) {
      std::cerr << "Vector sizes do not match for an inner product"
                << std::endl;
      return -1;
    }
    double result = 0.0;
    for (int i = 0; i < size; i++) {
      result += values[i] * other(i);
    }
    return result;
  }

  densemat outer(densevec other);

  void free() { delete[] values; }
};
densevec operator*(double k, densevec other) { return other * k; }

class densemat {
  int size = 0;
  double *values;

public:
  void clear() { std::memset(values, 0, sizeof(double) * size * size); }
  densemat() { values = new double[size * size]; }
  double *operator&() { return &values[0]; }
  void free() { delete[] values; }
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
  densemat operator+=(densemat other) {
    if (size != other.size) {
      std::cerr << "Matrix sizes do not match for an addition. Left is " << size
                << ", while right is " << other.getsize() << std::endl;
      return densemat();
    }
    for (int i = 0; i < size * size; i++) {
      values[i] += other(i / size, i % size);
    }
    return *this;
  }
  // Following operators for matrices
  // mat * scalar -> scalar. eltwise
  // scalar * mat -> scalar. eltwise
  // mat * mat -> mat. gemm
  // mat * vec -> vec. gemv
  // vec * mat -> vec. gemv
  //
  densemat operator*(double scalar) {
    std::vector<double> res_data;
    for (int i = 0; i < size * size; i++) {
      res_data.push_back(scalar * values[i]);
    }
    densemat result = densemat(res_data);
    return result;
  }
  densemat operator*(densemat other) {
    if (size != other.size) {
      std::cerr << "Matrix sizes do not match for a matrix product"
                << std::endl;
      return densemat();
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
  // Row of matrix shows up in result, column is contracted with another vector
  densevec operator*(densevec other) {
    if (size != other.getsize()) {
      std::cerr
          << "Matrix and vector sizes do not match for a matrix-vector product"
          << std::endl;
      return densevec();
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
densemat densevec::outer(densevec other) {
  std::vector<double> res_data;
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < other.size; j++) {
      res_data.push_back(values[i] * other(j));
    }
  }
  densemat result = densemat(res_data);
  return result;
}


densemat operator*(double k, densemat other) { return other * k; }
densevec operator*(densevec vec, densemat mat) { return mat * vec; }

#endif
