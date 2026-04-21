#pragma once
#include "matrix.hpp"
#include <random>
#include <vector>

namespace nn {

class Linear {
public:
    Matrix W, b;
    Matrix dW, db;
    Matrix last_input;

    Linear(int in_features, int out_features) {
        W  = Matrix::kaiming(in_features, out_features);
        b  = Matrix::zeros(1, out_features);
        dW = Matrix::zeros(in_features, out_features);
        db = Matrix::zeros(1, out_features);
    }

    Matrix forward(const Matrix& x) {
        last_input = x;
        return x.matmul(W).add_bias(b);
    }

    // Returns gradient w.r.t. input
    Matrix backward(const Matrix& grad_out) {
        // grad_out: (batch x out_features)
        dW = last_input.transpose().matmul(grad_out);
        db = grad_out.sum_axis0();
        return grad_out.matmul(W.transpose());
    }

    void zero_grad() { dW.zero_grad(); db.zero_grad(); }
};

class BatchNorm {
public:
    Matrix gamma, beta;
    Matrix d_gamma, d_beta;
    Matrix norm_cache, std_cache;
    float eps;

    BatchNorm(int features, float eps = 1e-5f) : eps(eps) {
        gamma   = Matrix::zeros(1, features);
        beta    = Matrix::zeros(1, features);
        d_gamma = Matrix::zeros(1, features);
        d_beta  = Matrix::zeros(1, features);
        for (int j = 0; j < features; ++j) { gamma.at(0, j) = 1.0f; }
    }

    Matrix forward(const Matrix& x) {
        int N = x.rows, C = x.cols;
        // mean per feature
        Matrix mean(1, C, 0.0f);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < C; ++j) mean.at(0, j) += x.at(i, j) / N;

        // variance per feature
        Matrix var(1, C, 0.0f);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < C; ++j) {
                float d = x.at(i, j) - mean.at(0, j);
                var.at(0, j) += d * d / N;
            }

        std_cache = var.apply([this](float v) { return std::sqrt(v + eps); });
        norm_cache = Matrix(N, C);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < C; ++j)
                norm_cache.at(i, j) = (x.at(i, j) - mean.at(0, j)) / std_cache.at(0, j);

        Matrix out(N, C);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < C; ++j)
                out.at(i, j) = gamma.at(0, j) * norm_cache.at(i, j) + beta.at(0, j);
        return out;
    }

    Matrix backward(const Matrix& dout) {
        int N = dout.rows, C = dout.cols;
        d_gamma.zero_grad(); d_beta.zero_grad();
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < C; ++j) {
                d_gamma.at(0, j) += dout.at(i, j) * norm_cache.at(i, j);
                d_beta.at(0, j)  += dout.at(i, j);
            }

        Matrix dnorm(N, C);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < C; ++j)
                dnorm.at(i, j) = dout.at(i, j) * gamma.at(0, j);

        Matrix dx(N, C, 0.0f);
        for (int j = 0; j < C; ++j) {
            float sum_dnorm = 0, sum_dnorm_x = 0;
            for (int i = 0; i < N; ++i) {
                sum_dnorm   += dnorm.at(i, j);
                sum_dnorm_x += dnorm.at(i, j) * norm_cache.at(i, j);
            }
            for (int i = 0; i < N; ++i) {
                dx.at(i, j) = (1.0f / (N * std_cache.at(0, j))) *
                    (N * dnorm.at(i, j) - sum_dnorm - norm_cache.at(i, j) * sum_dnorm_x);
            }
        }
        return dx;
    }

    void zero_grad() { d_gamma.zero_grad(); d_beta.zero_grad(); }
};

class Dropout {
public:
    float p;
    Matrix mask;
    bool training;

    Dropout(float p = 0.5f) : p(p), training(true) {}

    Matrix forward(const Matrix& x) {
        if (!training || p == 0.0f) return x;
        mask = Matrix(x.rows, x.cols);
        std::mt19937 rng(std::random_device{}());
        std::bernoulli_distribution dist(1.0f - p);
        float scale = 1.0f / (1.0f - p);
        for (int i = 0; i < (int)mask.data.size(); ++i)
            mask.data[i] = dist(rng) ? scale : 0.0f;
        return x.hadamard(mask);
    }

    Matrix backward(const Matrix& grad) {
        if (!training || p == 0.0f) return grad;
        return grad.hadamard(mask);
    }
};

} // namespace nn
