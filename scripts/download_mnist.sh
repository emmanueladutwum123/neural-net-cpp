#!/usr/bin/env bash
# Download and extract MNIST dataset
set -euo pipefail

DEST="data/mnist"
mkdir -p "$DEST"
BASE="http://yann.lecun.com/exdb/mnist"
MIRROR="https://storage.googleapis.com/cvdf-datasets/mnist"

FILES=(
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)

for f in "${FILES[@]}"; do
    if [ ! -f "$DEST/${f%.gz}" ]; then
        echo "Downloading $f..."
        curl -fL "$MIRROR/$f" -o "$DEST/$f" || curl -fL "$BASE/$f" -o "$DEST/$f"
        gunzip "$DEST/$f"
        echo "  -> $DEST/${f%.gz}"
    else
        echo "Already exists: $DEST/${f%.gz}"
    fi
done

echo "MNIST ready in $DEST/"
