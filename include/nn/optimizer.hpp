#pragma once
#include "matrix.hpp"
#include <vector>
#include <cmath>

namespace nn {

struct ParamRef {
    Matrix* param;
    Matrix* grad;
};

class Adam {
public:
    float lr, beta1, beta2, eps, weight_decay;
    int t = 0;
    std::vector<Matrix> m, v; // first and second moment estimates

    Adam(float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f,
         float eps = 1e-8f, float weight_decay = 0.0f)
        : lr(lr), beta1(beta1), beta2(beta2), eps(eps), weight_decay(weight_decay) {}

    void init(const std::vector<ParamRef>& params) {
        m.clear(); v.clear();
        for (auto& p : params) {
            m.push_back(Matrix::zeros(p.param->rows, p.param->cols));
            v.push_back(Matrix::zeros(p.param->rows, p.param->cols));
        }
    }

    void step(std::vector<ParamRef>& params) {
        ++t;
        float bc1 = 1.0f - std::pow(beta1, t);
        float bc2 = 1.0f - std::pow(beta2, t);

        for (int i = 0; i < (int)params.size(); ++i) {
            Matrix& g  = *params[i].grad;
            Matrix& p  = *params[i].param;

            if (weight_decay > 0.0f)
                for (int k = 0; k < (int)g.data.size(); ++k)
                    g.data[k] += weight_decay * p.data[k];

            for (int k = 0; k < (int)g.data.size(); ++k) {
                m[i].data[k] = beta1 * m[i].data[k] + (1.0f - beta1) * g.data[k];
                v[i].data[k] = beta2 * v[i].data[k] + (1.0f - beta2) * g.data[k] * g.data[k];
                float m_hat  = m[i].data[k] / bc1;
                float v_hat  = v[i].data[k] / bc2;
                p.data[k]   -= lr * m_hat / (std::sqrt(v_hat) + eps);
            }
        }
    }
};

class SGD {
public:
    float lr, momentum, weight_decay;
    std::vector<Matrix> velocity;

    SGD(float lr = 0.01f, float momentum = 0.9f, float weight_decay = 1e-4f)
        : lr(lr), momentum(momentum), weight_decay(weight_decay) {}

    void init(const std::vector<ParamRef>& params) {
        velocity.clear();
        for (auto& p : params)
            velocity.push_back(Matrix::zeros(p.param->rows, p.param->cols));
    }

    void step(std::vector<ParamRef>& params) {
        for (int i = 0; i < (int)params.size(); ++i) {
            Matrix& g = *params[i].grad;
            Matrix& p = *params[i].param;

            for (int k = 0; k < (int)g.data.size(); ++k) {
                float grad_k = g.data[k] + weight_decay * p.data[k];
                velocity[i].data[k] = momentum * velocity[i].data[k] + grad_k;
                p.data[k] -= lr * velocity[i].data[k];
            }
        }
    }
};

} // namespace nn
