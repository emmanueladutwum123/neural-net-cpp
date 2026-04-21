#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include "nn/matrix.hpp"
#include "nn/layers.hpp"
#include "nn/activations.hpp"

using namespace nn;

void benchmark_matmul() {
    std::cout << "=== Matrix Multiply Benchmark ===\n";
    for (int n : {64, 128, 256, 512, 1024}) {
        Matrix a = Matrix::randn(n, n);
        Matrix b = Matrix::randn(n, n);
        auto t0 = std::chrono::high_resolution_clock::now();
        int reps = std::max(1, 1000000 / (n * n * n));
        for (int i = 0; i < reps; ++i) volatile auto c = a.matmul(b);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
        double gflops = 2.0 * n * n * n / (ms * 1e6);
        std::cout << "  " << std::setw(4) << n << "x" << n
                  << "  " << std::setprecision(3) << ms << " ms"
                  << "  " << gflops << " GFLOPS\n";
    }
}

void benchmark_forward_pass() {
    std::cout << "\n=== MLP Forward Pass Benchmark (784->512->256->10) ===\n";
    Linear fc1(784, 512), fc2(512, 256), fc3(256, 10);
    ReLU relu;

    for (int bs : {1, 32, 128, 512}) {
        Matrix x = Matrix::randn(bs, 784);
        int reps = 1000;
        auto t0  = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < reps; ++i) {
            auto h1 = relu.forward(fc1.forward(x));
            auto h2 = relu.forward(fc2.forward(h1));
            volatile auto out = fc3.forward(h2);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
        std::cout << "  batch=" << std::setw(3) << bs
                  << "  " << std::setprecision(3) << ms << " ms/forward\n";
    }
}

int main() {
    benchmark_matmul();
    benchmark_forward_pass();
    return 0;
}
