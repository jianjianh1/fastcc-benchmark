#ifndef READ_HPP
#define READ_HPP
#include "contract.hpp"
#include <fstream>
#include <iostream>
#include <string>

template <> Tensor<float>::Tensor(std::string filename, bool has_header) {
  std::cout << "Reading tensor from file: " << filename << std::endl;
  std::ifstream file(filename);
  std::string line;
  int dimensionality = 0;
  std::vector<int> extents;
  while (std::getline(file, line)) {
    std::vector<int> nnz_data;
    if (!std::isdigit(line[0])) {
      continue;
    }
    if (has_header) {
      has_header = false;
      continue;
    }
    auto pos = line.find(" ");
    while (pos != std::string::npos) {
      nnz_data.push_back(std::stoi(line.substr(0, pos)));
      line = line.substr(pos + 1);
      pos = line.find(" ");
    }
    if (dimensionality == 0) {
      dimensionality = nnz_data.size();
      std::fill_n(std::back_inserter(extents), dimensionality, INT_MIN);
    } else {
      for (int i = 0; i < dimensionality; i++) {
        if (extents[i] < nnz_data[i]) {
          extents[i] = nnz_data[i];
        }
      }
      assert(dimensionality == nnz_data.size());
    }
    float nnz_val = std::stod(line.substr(0, pos));
    CoOrdinate this_coords = CoOrdinate(nnz_data);
    NNZ<float> this_nnz = NNZ<float>(nnz_val, this_coords);
    this->nonzeros.push_back(this_nnz);
  }
  this->dimensionality = dimensionality;
  this->shape = new int[dimensionality];
  memcpy(this->shape, extents.data(), dimensionality * sizeof(int));

  /** TODO remove before flight **/
  std::cout << "Tensor shape: ";
  for (int i = 0; i < dimensionality; i++) {
    std::cout << this->shape[i] << " ";
  }
  std::cout << std::endl;
  std::cout << "Tensor nnz count: " << this->nonzeros.size() << std::endl;
}

template <> Tensor<densevec>::Tensor(std::string filename, bool has_header) {
  std::ifstream file(filename);
  Tensor<float> eltwise = Tensor<float>(filename, has_header);
  for (int base = 0; base < eltwise.get_nonzeros().size(); base++) {
    auto base_coords = eltwise.get_nonzeros()[base].get_coords();
    auto base_data = eltwise.get_nonzeros()[base].get_data();
    auto positions_to_remove =
        CoOrdinate({base_coords.get_dimensionality() - 1});
    auto base_coords_outer = base_coords.remove(positions_to_remove);
    std::vector<double> vec_data;
    int bound = base;
    while (bound < eltwise.get_nonzeros().size() &&
           base_coords_outer ==
               eltwise.get_nonzeros()[bound].get_coords().remove(
                   positions_to_remove)) {
      vec_data.push_back(eltwise.get_nonzeros()[bound].get_data());
      bound++;
    }
    densevec vec = densevec(vec_data);
    nonzeros.emplace_back(vec, base_coords_outer);
    base = bound - 1;
  }
  this->_infer_dimensionality();
  this->_infer_shape();
}

template <> Tensor<densemat>::Tensor(std::string filename, bool has_header) {
  std::ifstream file(filename);
  Tensor<float> eltwise = Tensor<float>(filename, has_header);
  // Tensor<densevec> result = Tensor<densevec>(eltwise.get_size());
  for (int base = 0; base < eltwise.get_nonzeros().size(); base++) {
    auto base_coords = eltwise.get_nonzeros()[base].get_coords();
    auto base_data = eltwise.get_nonzeros()[base].get_data();
    auto positions_to_remove =
        CoOrdinate({base_coords.get_dimensionality() - 1,
                    base_coords.get_dimensionality() - 2});
    auto base_coords_outer = base_coords.remove(positions_to_remove);
    std::vector<double> vec_data;
    int bound = base;
    while (bound < eltwise.get_nonzeros().size() &&
           base_coords_outer ==
               eltwise.get_nonzeros()[bound].get_coords().remove(
                   positions_to_remove)) {
      vec_data.push_back(eltwise.get_nonzeros()[bound].get_data());
      bound++;
    }
    densemat mat = densemat(vec_data);
    nonzeros.emplace_back(mat, base_coords_outer);
    base = bound - 1;
  }
  this->_infer_dimensionality();
  this->_infer_shape();
}

template <> void Tensor<double>::write(std::string filename) {
  std::ofstream file(filename, std::ios_base::app);
  for (int i = 0; i < this->dimensionality; i++) {
    file << this->shape[i] << " ";
  }
  file << std::endl;
  for (auto &nnz : this->nonzeros) {
    file << nnz.get_coords().to_string() << " " << nnz.get_data() << std::endl;
  }
  file.close();
}
template <class DT> void Tensor<DT>::write(std::string filename) {
  std::ofstream file(filename, std::ios_base::app);
  for (int i = 0; i < this->dimensionality; i++) {
    file << this->shape[i] << " ";
  }
  file << std::endl;
  for (auto &nnz : this->nonzeros) {
    file << nnz.get_coords().to_string() << " " << nnz.get_data().to_string()
         << std::endl;
  }
  file.close();
}
std::string CoOrdinate::to_string() const {
  std::string str = "";
  for (int i = 0; i < this->coords.size(); i++) {
    str += std::to_string(this->coords[i]) + " ";
  }
  return str;
}

void CoOrdinate::write(std::string filename) const {
  std::ofstream file(filename, std::ios_base::app);
  file << this->to_string() << std::endl;
  file << std::endl;
  file.close();
}
#endif
