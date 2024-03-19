#ifndef READ_HPP
#define READ_HPP
#include "contract.hpp"
#include <fstream>
#include <iostream>
#include <string>

Tensor::Tensor(std::string filename, bool has_header = false) {
    std::cout<<"Reading tensor from file: "<<filename<<std::endl;
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
    NNZ this_nnz = NNZ(nnz_val, this_coords);
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
void Tensor::write(std::string filename) {
  std::ofstream file(filename);
  for (auto &nnz : this->nonzeros) {
    file << nnz.get_coords().to_string() << " " << nnz.get_data() << std::endl;
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
