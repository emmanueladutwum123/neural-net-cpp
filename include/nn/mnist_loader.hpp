#pragma once
#include "matrix.hpp"
#include <string>
#include <vector>

namespace nn {

struct MNISTData {
    Matrix images; // (N, 784) normalized [0,1]
    std::vector<int> labels; // class indices 0-9
};

MNISTData load_mnist(const std::string& images_path, const std::string& labels_path);

} // namespace nn
