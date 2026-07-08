#include "nn/mnist_loader.hpp"
#include <fstream>
#include <stdexcept>

namespace nn {

static int read_int(std::ifstream& f) {
    unsigned char buf[4];
    f.read(reinterpret_cast<char*>(buf), 4);
    return (buf[0] << 24) | (buf[1] << 16) | (buf[2] << 8) | buf[3];
}

MNISTData load_mnist(const std::string& images_path, const std::string& labels_path) {
    std::ifstream img_f(images_path, std::ios::binary);
    std::ifstream lbl_f(labels_path, std::ios::binary);

    if (!img_f) throw std::runtime_error("Cannot open: " + images_path);
    if (!lbl_f) throw std::runtime_error("Cannot open: " + labels_path);

    int magic_img = read_int(img_f);
    int num_imgs  = read_int(img_f);
    int rows      = read_int(img_f);
    int cols      = read_int(img_f);
    if (magic_img != 0x00000803) throw std::runtime_error("Invalid MNIST image magic");

    int magic_lbl = read_int(lbl_f);
    int num_lbls  = read_int(lbl_f);
    if (magic_lbl != 0x00000801) throw std::runtime_error("Invalid MNIST label magic");
    if (num_imgs != num_lbls)    throw std::runtime_error("Image/label count mismatch");

    int n_pixels = rows * cols;
    MNISTData data;
    data.images = Matrix(num_imgs, n_pixels);
    data.labels.resize(num_imgs);

    std::vector<unsigned char> buf(n_pixels);
    for (int i = 0; i < num_imgs; ++i) {
        img_f.read(reinterpret_cast<char*>(buf.data()), n_pixels);
        for (int p = 0; p < n_pixels; ++p)
            data.images.at(i, p) = buf[p] / 255.0f;
        data.labels[i] = static_cast<int>(static_cast<unsigned char>(lbl_f.get()));
    }

    return data;
}

} // namespace nn
