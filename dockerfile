FROM nvidia/cuda:12.3.1-devel-ubuntu22.04 AS base

WORKDIR /app

# Install System Dependencies
# fftw3-dev: CPU convolution fallback
# cmake/git: Build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libfftw3-dev \
    libtbb-dev \
    pkg-config \
    python3 \
    python3-pip \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir numpy scipy matplotlib soundfile trimesh

# We stop here for development. Source code is NOT copied; it will be mounted live.
FROM base AS dev
# (Optional) Add any dev-specific tools here like gdb, clang-format, etc.

# This stage copies the source and builds the binary into the image.
FROM base AS deploy
COPY . /app
RUN mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

CMD ["./build/acoustic_sim"]