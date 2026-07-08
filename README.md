# Neural Network from Scratch — C++17

A deep learning framework implemented in pure C++17 — no external ML libraries. A 3-layer MLP
trained with Adam and cosine LR decay reaches **98.5% test accuracy on real MNIST** (verified by
actually running it to completion — see [Sample Output](#sample-output) below).

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
| **Testing** | Google Test — component tests plus an end-to-end training regression test |
| **CI/CD** | GitHub Actions — build + unit tests + benchmark + a real MNIST training smoke test on every push |

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

# Train (20 epochs by default; optional second argument overrides epoch count,
# e.g. `build/train data/mnist 2` for a quick run)
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

13 tests: matrix ops, activations (including a direct `ReLU::backward` check), `Linear` shapes,
cross-entropy, Adam parameter updates, dropout, and one end-to-end integration test that runs a
real `MLP` through several forward + backward + optimizer steps and asserts the loss actually goes
down — see [What Was Fixed](#what-was-fixed) for why that last one exists.

## Sample Output

Real, unedited output from `build/train data/mnist` (Apple M2, `-O3`, Release build):

```
Loading MNIST...
Train: 60000 | Test: 10000
Epoch  1/20  loss=0.2428  train_acc=97.6150%  test_acc=96.9600%  lr=0.0010  time=8.7s
Epoch  2/20  loss=0.1066  train_acc=98.5067%  test_acc=97.4000%  lr=0.0010  time=8.4s
Epoch  3/20  loss=0.0788  train_acc=98.9017%  test_acc=97.6200%  lr=0.0009  time=8.6s
Epoch  4/20  loss=0.0659  train_acc=99.2400%  test_acc=97.9500%  lr=0.0009  time=8.3s
Epoch  5/20  loss=0.0520  train_acc=99.3200%  test_acc=98.0300%  lr=0.0009  time=8.6s
Epoch  6/20  loss=0.0448  train_acc=99.4600%  test_acc=98.0200%  lr=0.0008  time=8.4s
Epoch  7/20  loss=0.0408  train_acc=99.5667%  test_acc=97.9700%  lr=0.0008  time=8.8s
Epoch  8/20  loss=0.0328  train_acc=99.6767%  test_acc=98.1500%  lr=0.0007  time=8.5s
Epoch  9/20  loss=0.0305  train_acc=99.7400%  test_acc=98.1700%  lr=0.0007  time=14.7s
Epoch 10/20  loss=0.0236  train_acc=99.8333%  test_acc=98.0200%  lr=0.0006  time=8.3s
Epoch 11/20  loss=0.0201  train_acc=99.8900%  test_acc=98.2700%  lr=0.0005  time=8.9s
Epoch 12/20  loss=0.0158  train_acc=99.9183%  test_acc=98.3000%  lr=0.0004  time=9.0s
Epoch 13/20  loss=0.0141  train_acc=99.9450%  test_acc=98.3300%  lr=0.0003  time=9.9s
Epoch 14/20  loss=0.0119  train_acc=99.9500%  test_acc=98.4500%  lr=0.0003  time=9.3s
Epoch 15/20  loss=0.0095  train_acc=99.9900%  test_acc=98.4700%  lr=0.0002  time=15.1s
Epoch 16/20  loss=0.0084  train_acc=99.9917%  test_acc=98.4000%  lr=0.0001  time=10.7s
Epoch 17/20  loss=0.0076  train_acc=99.9933%  test_acc=98.4200%  lr=0.0001  time=8.8s
Epoch 18/20  loss=0.0062  train_acc=99.9950%  test_acc=98.4500%  lr=0.0001  time=8.7s
Epoch 19/20  loss=0.0066  train_acc=99.9967%  test_acc=98.5700%  lr=0.0000  time=8.7s
Epoch 20/20  loss=0.0057  train_acc=99.9967%  test_acc=98.5400%  lr=0.0000  time=8.4s
```

Final test accuracy: **98.54%** (peak 98.57% at epoch 19). ~8-9s/epoch, ~3 minutes total.

Benchmark output (same machine):

```
=== Matrix Multiply Benchmark ===
    64x64  0.0484 ms  10.8 GFLOPS
   128x128  0.386 ms  10.9 GFLOPS
   256x256  3.62 ms  9.26 GFLOPS
   512x512  17.8 ms  15.1 GFLOPS
  1024x1024  79.1 ms  27.2 GFLOPS

=== MLP Forward Pass Benchmark (784->512->256->10) ===
  batch=  1  0.0411 ms/forward
  batch= 32  1.34 ms/forward
  batch=128  5.3 ms/forward
  batch=512  21.4 ms/forward
```

The matmul is a plain triple-nested loop (row-major, no blocking/tiling beyond what `-O3`
auto-vectorizes) — GFLOPS climbing with matrix size here reflects better amortization of loop
overhead, not algorithmic improvement. A tiled/blocked implementation would be a natural next step
for meaningfully higher throughput.

## What Was Fixed

This project previously claimed ~97.8% MNIST accuracy that had never actually been produced by a
successful run. Two bugs were compounding:

1. `src/mnist_loader.cpp` included `"mnist_loader.hpp"` instead of `"nn/mnist_loader.hpp"` (every
   other file uses the `nn/`-prefixed path) — this alone failed the CMake build, which is why CI
   was red.
2. Once the build was fixed, `MLP::backward()` (then defined inline in `src/train.cpp`) called
   `relu.backward(g2, {})` — passing an empty, default-constructed `Matrix{}` instead of the real
   pre-activation input `ReLU::backward` needs to compute its gradient mask. `Matrix::hadamard()`
   asserts matching shapes, but this project builds in `Release` mode, where `assert()` is compiled
   out — so instead of failing loudly, it read out of bounds of an empty vector and **segfaulted on
   the first training step, every time**. None of the (all individually correct) unit tests caught
   this because every one of them tested a component in isolation; none drove the full
   `MLP::forward()` + `MLP::backward()` path.

The fix: `ReLU` now caches its own input in `forward()` (`last_input`, mirroring how `Linear`
already does this) instead of relying on a caller to thread the pre-activation through by hand —
`backward()` no longer takes it as a parameter at all, so there's no longer a slot for that value to
go missing from. `MLP` moved out of `train.cpp` into `include/nn/mlp.hpp` so it's an actual testable
unit, and `tests/test_nn.cpp` gained an integration test that runs a real `MLP` through several
training steps and asserts the loss decreases — the regression test that would have caught this
before it shipped. CI now also downloads real MNIST and runs a 2-epoch training smoke test on every
push, so a crash or an accuracy collapse fails CI instead of only a "does it compile" check passing
silently.

Also fixed while auditing every file: two headers used `std::max`/`std::fill` without `#include
<algorithm>` (worked only because it was transitively included elsewhere — not guaranteed);
`Dropout` was reseeding a fresh Mersenne Twister from `std::random_device` on every single
`forward()` call instead of once at construction; `CMakeLists.txt` had `-march=native` and
`-ffast-math`, both dropped for portability and numerical predictability (`-O3` alone is what the
benchmark numbers above reflect).

## Key Implementation Highlights

- **Zero-dependency matrix engine** with cache-friendly row-major layout, Kaiming/Xavier init, hadamard product, axis-sum, and full matmul
- **Numerically stable softmax** with max subtraction before exp
- **Backpropagation** through Linear, BatchNorm, Dropout, ReLU with correct gradient formulas, verified end-to-end by an integration test, not just unit tests of each piece
- **Adam optimizer** with bias correction and optional weight decay (AdamW)
- **Cosine LR annealing** computed per epoch without external scheduler

## Tech Stack

C++17 · CMake · Google Test · GitHub Actions
