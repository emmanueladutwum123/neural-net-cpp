#pragma once
#include "matrix.hpp"
#include <cmath>

namespace nn {

struct ReLU {
    Matrix forward(const Matrix& x) {
        return x.apply([](float v) { return v > 0.0f ? v : 0.0f; });
    }
    Matrix backward(const Matrix& grad, const Matrix& x) {
        return grad.hadamard(x.apply([](float v) { return v > 0.0f ? 1.0f : 0.0f; }));
    }
};

struct GELU {
    // Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    Matrix forward(const Matrix& x) {
        return x.apply([](float v) {
            float c = 0.7978845608f; // sqrt(2/pi)
            float inner = c * (v + 0.044715f * v * v * v);
            return 0.5f * v * (1.0f + std::tanh(inner));
        });
    }
    Matrix backward(const Matrix& grad, const Matrix& x) {
        return grad.hadamard(x.apply([](float v) {
            float c = 0.7978845608f;
            float inner = c * (v + 0.044715f * v * v * v);
            float tanh_val = std::tanh(inner);
            float sech2 = 1.0f - tanh_val * tanh_val;
            float dtanh = c * (1.0f + 3.0f * 0.044715f * v * v);
            return 0.5f * (1.0f + tanh_val) + 0.5f * v * sech2 * dtanh;
        }));
    }
};

struct Sigmoid {
    Matrix forward(const Matrix& x) {
        return x.apply([](float v) { return 1.0f / (1.0f + std::exp(-v)); });
    }
    Matrix backward(const Matrix& grad, const Matrix& out) {
        // out is sigmoid output
        return grad.hadamard(out.hadamard(out.apply([](float v) { return 1.0f - v; })));
    }
};

struct Softmax {
    Matrix forward(const Matrix& x) {
        Matrix result(x.rows, x.cols);
        for (int i = 0; i < x.rows; ++i) {
            float max_val = x.at(i, 0);
            for (int j = 1; j < x.cols; ++j) max_val = std::max(max_val, x.at(i, j));
            float sum = 0.0f;
            for (int j = 0; j < x.cols; ++j) {
                result.at(i, j) = std::exp(x.at(i, j) - max_val);
                sum += result.at(i, j);
            }
            for (int j = 0; j < x.cols; ++j) result.at(i, j) /= sum;
        }
        return result;
    }
};

} // namespace nn
