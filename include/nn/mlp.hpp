#pragma once
#include "activations.hpp"
#include "layers.hpp"
#include "matrix.hpp"
#include "optimizer.hpp"
#include <vector>

namespace nn {

// 3-layer MLP: 784 -> 512 -> 256 -> 10
struct MLP {
    Linear fc1{784, 512}, fc2{512, 256}, fc3{256, 10};
    BatchNorm bn1{512}, bn2{256};
    Dropout drop1{0.2f}, drop2{0.2f};
    // Two independent instances (not one shared ReLU) — each forward() call caches
    // its own pre-activation input, so backward() must read it back from the same
    // instance that produced it, not from whichever layer happened to run last.
    ReLU relu1, relu2;

    Matrix forward(const Matrix& x, bool training = true) {
        drop1.training = drop2.training = training;
        auto h1 = relu1.forward(bn1.forward(fc1.forward(x)));
        h1 = drop1.forward(h1);
        auto h2 = relu2.forward(bn2.forward(fc2.forward(h1)));
        h2 = drop2.forward(h2);
        return fc3.forward(h2);
    }

    Matrix backward(const Matrix& grad_out) {
        auto g3 = fc3.backward(grad_out);
        auto g2 = drop2.backward(g3);
        auto gb2 = bn2.backward(relu2.backward(g2));
        auto gf2 = fc2.backward(gb2);
        auto g1 = drop1.backward(gf2);
        auto gb1 = bn1.backward(relu1.backward(g1));
        return fc1.backward(gb1);
    }

    std::vector<ParamRef> params() {
        return {
            {&fc1.W, &fc1.dW}, {&fc1.b, &fc1.db},
            {&bn1.gamma, &bn1.d_gamma}, {&bn1.beta, &bn1.d_beta},
            {&fc2.W, &fc2.dW}, {&fc2.b, &fc2.db},
            {&bn2.gamma, &bn2.d_gamma}, {&bn2.beta, &bn2.d_beta},
            {&fc3.W, &fc3.dW}, {&fc3.b, &fc3.db},
        };
    }

    void zero_grad() {
        fc1.zero_grad(); fc2.zero_grad(); fc3.zero_grad();
        bn1.zero_grad(); bn2.zero_grad();
    }
};

} // namespace nn
