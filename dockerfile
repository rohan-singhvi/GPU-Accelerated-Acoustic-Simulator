# 1. Base Stage: Install Dependencies
FROM nvidia/cuda:12.3.1-devel-ubuntu22.04 AS base

WORKDIR /app

# Install System Dependencies
# - C++: build-essential, cmake, git, libfftw3-dev
# - Python: python3, python3-pip (for helper scripts)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libfftw3-dev \
    pkg-config \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python Helper Dependencies
# We use standard pip here since it's just for helper scripts, not the main engine
RUN pip3 install --no-cache-dir numpy scipy soundfile trimesh

# 2. Dev Stage: For VS Code / DevContainer
FROM base AS dev
# Source code is NOT copied here; it will be mounted live by DevContainer.

# 3. Deploy Stage: For Production / Standalone
FROM base AS deploy
COPY . /app
RUN mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

CMD ["./build/acoustic_sim"]