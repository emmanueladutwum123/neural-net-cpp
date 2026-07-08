#pragma once
#include "matrix.hpp"
#include "activations.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace nn {

struct CrossEntropyLoss {
    Matrix probs;

    // x: logits (batch x classes), labels: vector of class indices
    float forward(const Matrix& x, const std::vector<int>& labels) {
        Softmax sm;
        probs = sm.forward(x);
        float loss = 0.0f;
        for (int i = 0; i < (int)labels.size(); ++i)
            loss -= std::log(std::max(probs.at(i, labels[i]), 1e-9f));
        return loss / labels.size();
    }

    // Returns gradient w.r.t. logits
    Matrix backward(const std::vector<int>& labels) {
        Matrix grad = probs;
        float N = labels.size();
        for (int i = 0; i < (int)labels.size(); ++i)
            grad.at(i, labels[i]) -= 1.0f;
        for (auto& v : grad.data) v /= N;
        return grad;
    }
};

struct MSELoss {
    Matrix diff;

    float forward(const Matrix& pred, const Matrix& target) {
        diff = pred - target;
        float loss = 0.0f;
        for (auto v : diff.data) loss += v * v;
        return loss / diff.data.size();
    }

    Matrix backward() {
        return diff * (2.0f / diff.data.size());
    }
};

} // namespace nn
