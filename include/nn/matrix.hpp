#pragma once
#include <vector>
#include <stdexcept>
#include <cmath>
#include <random>
#include <functional>
#include <cassert>

namespace nn {

class Matrix {
public:
    int rows, cols;
    std::vector<float> data;

    Matrix() : rows(0), cols(0) {}
    Matrix(int r, int c, float val = 0.0f) : rows(r), cols(c), data(r * c, val) {}

    float& at(int r, int c) { return data[r * cols + c]; }
    const float& at(int r, int c) const { return data[r * cols + c]; }

    static Matrix zeros(int r, int c) { return Matrix(r, c, 0.0f); }

    static Matrix randn(int r, int c, float mean = 0.0f, float std = 1.0f) {
        Matrix m(r, c);
        std::mt19937 rng(std::random_device{}());
        std::normal_distribution<float> dist(mean, std);
        for (auto& v : m.data) v = dist(rng);
        return m;
    }

    static Matrix xavier(int r, int c) {
        float std = std::sqrt(2.0f / (r + c));
        return randn(r, c, 0.0f, std);
    }

    static Matrix kaiming(int r, int c) {
        float std = std::sqrt(2.0f / r);
        return randn(r, c, 0.0f, std);
    }

    Matrix operator+(const Matrix& o) const {
        assert(rows == o.rows && cols == o.cols);
        Matrix result(rows, cols);
        for (int i = 0; i < (int)data.size(); ++i) result.data[i] = data[i] + o.data[i];
        return result;
    }

    Matrix operator-(const Matrix& o) const {
        assert(rows == o.rows && cols == o.cols);
        Matrix result(rows, cols);
        for (int i = 0; i < (int)data.size(); ++i) result.data[i] = data[i] - o.data[i];
        return result;
    }

    Matrix operator*(float s) const {
        Matrix result(rows, cols);
        for (int i = 0; i < (int)data.size(); ++i) result.data[i] = data[i] * s;
        return result;
    }

    Matrix hadamard(const Matrix& o) const {
        assert(rows == o.rows && cols == o.cols);
        Matrix result(rows, cols);
        for (int i = 0; i < (int)data.size(); ++i) result.data[i] = data[i] * o.data[i];
        return result;
    }

    // Matrix multiply: this (m x k) @ o (k x n) -> (m x n)
    Matrix matmul(const Matrix& o) const {
        assert(cols == o.rows);
        Matrix result(rows, o.cols, 0.0f);
        for (int i = 0; i < rows; ++i)
            for (int k = 0; k < cols; ++k) {
                float a = at(i, k);
                for (int j = 0; j < o.cols; ++j)
                    result.at(i, j) += a * o.at(k, j);
            }
        return result;
    }

    Matrix transpose() const {
        Matrix result(cols, rows);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result.at(j, i) = at(i, j);
        return result;
    }

    // Broadcast bias addition: this (batch x n) + bias (1 x n)
    Matrix add_bias(const Matrix& bias) const {
        assert(cols == bias.cols && bias.rows == 1);
        Matrix result(rows, cols);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result.at(i, j) = at(i, j) + bias.at(0, j);
        return result;
    }

    void apply_inplace(std::function<float(float)> fn) {
        for (auto& v : data) v = fn(v);
    }

    Matrix apply(std::function<float(float)> fn) const {
        Matrix result(rows, cols);
        for (int i = 0; i < (int)data.size(); ++i) result.data[i] = fn(data[i]);
        return result;
    }

    // Sum along axis 0 -> (1 x cols)
    Matrix sum_axis0() const {
        Matrix result(1, cols, 0.0f);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                result.at(0, j) += at(i, j);
        return result;
    }

    float sum() const {
        float s = 0;
        for (auto v : data) s += v;
        return s;
    }

    float mean() const { return sum() / data.size(); }

    void zero_grad() { std::fill(data.begin(), data.end(), 0.0f); }

    void operator+=(const Matrix& o) {
        for (int i = 0; i < (int)data.size(); ++i) data[i] += o.data[i];
    }

    void operator-=(const Matrix& o) {
        for (int i = 0; i < (int)data.size(); ++i) data[i] -= o.data[i];
    }
};

} // namespace nn
