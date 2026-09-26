#!/usr/bin/env bash
# Build and test the C++ helpers (hx-search, later hx-exec) into cpp/build.
set -euo pipefail
cd "$(dirname "$0")/.."
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE="${BUILD_TYPE:-Release}"
cmake --build cpp/build -j
(cd cpp/build && ctest --output-on-failure)
