# Neural Network from Scratch — C++17

A production-quality deep learning framework implemented in pure C++17 — no external ML libraries. Achieves **~97% accuracy on MNIST** with a 3-layer MLP trained using Adam optimizer with cosine LR decay.

## Architecture

```
Input (784) → Linear(512) → BatchNorm → ReLU → Dropout(0.2)
            → Linear(256) → BatchNorm → ReLU → Dropout(0.2)
            → Linear(10)  → Softmax → CrossEntropyLoss
```

## Features

| Component | Details |
|-----------|---------|
| **Layers** | Linear, BatchNorm, Dropout |
| **Activations** | ReLU, GELU, Sigmoid, Softmax |
| **Optimizers** | Adam (β₁=0.9, β₂=0.999), SGD with momentum |
| **Loss** | Cross-Entropy, MSE |
| **LR Schedule** | Cosine annealing |
| **Weight Init** | Kaiming (ReLU layers), Xavier |
| **Testing** | Google Test — matrix ops, forward/backward, optimizer |
| **CI/CD** | GitHub Actions — build + test + benchmark on every push |

## Build

```bash
# Requires CMake 3.16+, g++/clang++ with C++17, optional Google Test
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

## Train on MNIST

```bash
# Download MNIST binary files
bash scripts/download_mnist.sh

# Train (20 epochs, ~97% test accuracy)
build/train data/mnist
```

## Benchmark

```bash
build/benchmark
# Outputs matrix multiply GFLOPS and MLP forward-pass latency per batch size
```

## Run Tests

```bash
build/test_nn
```

## Key Implementation Highlights

- **Zero-dependency matrix engine** with cache-friendly row-major layout, Kaiming/Xavier init, hadamard product, axis-sum, and full matmul
- **Numerically stable softmax** with max subtraction before exp
- **Backpropagation** through Linear, BatchNorm, Dropout with correct gradient formulas
- **Adam optimizer** with bias correction and optional weight decay (AdamW)
- **Cosine LR annealing** computed per epoch without external scheduler

## Tech Stack

C++17 · CMake · Google Test · GitHub Actions
