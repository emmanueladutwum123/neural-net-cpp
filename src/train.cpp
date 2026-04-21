#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iomanip>

#include "nn/matrix.hpp"
#include "nn/layers.hpp"
#include "nn/activations.hpp"
#include "nn/optimizer.hpp"
#include "nn/loss.hpp"
#include "nn/mnist_loader.hpp"

using namespace nn;

// 3-layer MLP: 784 -> 512 -> 256 -> 10
struct MLP {
    Linear fc1{784, 512}, fc2{512, 256}, fc3{256, 10};
    BatchNorm bn1{512}, bn2{256};
    Dropout drop1{0.2f}, drop2{0.2f};
    ReLU relu;

    Matrix forward(const Matrix& x, bool training = true) {
        drop1.training = drop2.training = training;
        auto h1 = relu.forward(bn1.forward(fc1.forward(x)));
        h1 = drop1.forward(h1);
        auto h2 = relu.forward(bn2.forward(fc2.forward(h1)));
        h2 = drop2.forward(h2);
        return fc3.forward(h2);
    }

    Matrix backward(const Matrix& grad_out) {
        auto g3 = fc3.backward(grad_out);
        auto g2 = drop2.backward(g3);
        auto gb2 = bn2.backward(relu.backward(g2, {}));
        auto gf2 = fc2.backward(gb2);
        auto g1 = drop1.backward(gf2);
        auto gb1 = bn1.backward(relu.backward(g1, {}));
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

float evaluate(MLP& model, const Matrix& images, const std::vector<int>& labels, int batch_size = 512) {
    int correct = 0, n = images.rows;
    for (int start = 0; start < n; start += batch_size) {
        int end = std::min(start + batch_size, n);
        int bs = end - start;
        Matrix batch(bs, images.cols);
        for (int i = 0; i < bs; ++i)
            for (int j = 0; j < images.cols; ++j)
                batch.at(i, j) = images.at(start + i, j);

        auto logits = model.forward(batch, false);
        for (int i = 0; i < bs; ++i) {
            int pred = 0;
            for (int c = 1; c < logits.cols; ++c)
                if (logits.at(i, c) > logits.at(i, pred)) pred = c;
            if (pred == labels[start + i]) ++correct;
        }
    }
    return 100.0f * correct / n;
}

int main(int argc, char** argv) {
    std::string data_dir = argc > 1 ? argv[1] : "data/mnist";
    int epochs = 20, batch_size = 128;
    float lr = 1e-3f;

    std::cout << "Loading MNIST..." << std::endl;
    auto train_data = load_mnist(data_dir + "/train-images-idx3-ubyte",
                                  data_dir + "/train-labels-idx1-ubyte");
    auto test_data  = load_mnist(data_dir + "/t10k-images-idx3-ubyte",
                                  data_dir + "/t10k-labels-idx1-ubyte");
    std::cout << "Train: " << train_data.images.rows << " | Test: " << test_data.images.rows << "\n";

    MLP model;
    Adam optimizer(lr, 0.9f, 0.999f, 1e-8f, 1e-4f);
    auto param_refs = model.params();
    optimizer.init(param_refs);

    CrossEntropyLoss criterion;

    int n = train_data.images.rows;
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::shuffle(indices.begin(), indices.end(), std::mt19937(epoch));
        float epoch_loss = 0.0f;
        int steps = 0;

        // Cosine LR decay
        float cur_lr = lr * 0.5f * (1.0f + std::cos(M_PI * epoch / epochs));
        optimizer.lr = cur_lr;

        auto t0 = std::chrono::steady_clock::now();

        for (int start = 0; start < n; start += batch_size) {
            int end = std::min(start + batch_size, n);
            int bs = end - start;

            Matrix batch_x(bs, 784);
            std::vector<int> batch_y(bs);
            for (int i = 0; i < bs; ++i) {
                int idx = indices[start + i];
                for (int j = 0; j < 784; ++j)
                    batch_x.at(i, j) = train_data.images.at(idx, j);
                batch_y[i] = train_data.labels[idx];
            }

            auto logits = model.forward(batch_x, true);
            float loss  = criterion.forward(logits, batch_y);
            auto grad   = criterion.backward(batch_y);

            model.zero_grad();
            model.backward(grad);
            optimizer.step(param_refs);

            epoch_loss += loss;
            ++steps;
        }

        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();

        float train_acc = evaluate(model, train_data.images, train_data.labels);
        float test_acc  = evaluate(model, test_data.images,  test_data.labels);

        std::cout << std::fixed << std::setprecision(4)
                  << "Epoch " << std::setw(2) << epoch + 1 << "/" << epochs
                  << "  loss=" << epoch_loss / steps
                  << "  train_acc=" << train_acc << "%"
                  << "  test_acc="  << test_acc  << "%"
                  << "  lr=" << cur_lr
                  << "  time=" << std::setprecision(1) << secs << "s\n";
    }

    return 0;
}
