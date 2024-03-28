#ifndef TYPES_H
#define TYPES_H
#include <cstring>
#include <iostream>
#include <stdlib.h>
#include <vector>

//TODO implement identity value, and += operator (in-place)
class densevec {
  int size;
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
  double *operator&() { return &values[0]; }
  int getsize() { return size; }
  // Now we need to implement the following
  // vec * scalar
  // scalar * vec
  // vec * vec -> inner dot, we don't ever need  element-wise
  densevec operator*(double scalar) {
    densevec result = densevec();
    for (int i = 0; i < size; i++) {
      *(&result + i) = values[i] * scalar;
    }
    return result;
  }
  std::string to_string() {
    std::string result = "";
    for (int i = 0; i < size; i++) {
      result += std::to_string(values[i]) + " ";
    }
    return result;
  }
  double operator*(densevec other) {
    if (size != other.size) {
      std::cerr << "Vector sizes do not match for an inner product"
                << std::endl;
      return -1;
    }
    double result = 0.0;
    for (int i = 0; i < size; i++) {
      result += values[i] * *(&other + i);
    }
    return result;
  }
  void free() { delete[] values; }
};
densevec operator*(double k, densevec other) { return other * k; }

class densemat {
  int size;
  double *values;

public:
  void clear() { std::memset(values, 0, sizeof(double) * size * size); }
  densemat() { values = new double[size * size]; }
  double *operator&() { return &values[0]; }
  void free() { delete[] values; }
  int getsize() { return size; }
  // Following operators for matrices
  // mat * scalar -> scalar. eltwise
  // scalar * mat -> scalar. eltwise
  // mat * mat -> mat. gemm
  // mat * vec -> vec. gemv
  // vec * mat -> vec. gemv
  //
  densemat operator*(double scalar) {
    densemat result = densemat();
    for (int i = 0; i < size * size; i++) {
      *(&result + i) = scalar * values[i];
    }
    return result;
  }
  densemat operator*(densemat other) {
    if (size != other.size) {
      std::cerr << "Matrix sizes do not match for a matrix product"
                << std::endl;
      return densemat();
    }
    densemat result = densemat();
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < size; j++) {
        double sum = 0.0;
        for (int k = 0; k < size; k++) {
          sum += values[i * size + k] * *(&other + k * size + j);
        }
        *(&result + i * size + j) = sum;
      }
    }
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
    densevec result = densevec();
    for (int i = 0; i < size; i++) {
      double sum = 0.0;
      for (int j = 0; j < size; j++) {
        sum += values[i * size + j] * *(&other + j);
      }
      *(&result + i) = sum;
    }
    return result;
  }
};

densemat operator*(double k, densemat other) { return other * k; }
densevec operator*(densevec vec, densemat mat) { return mat * vec; }

#endif
