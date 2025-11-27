Understood. Here is the raw Markdown text for the `README.md`. You can copy the block below directly.

# GPU-Accelerated Acoustic Simulator (C++/CUDA)

A high-performance, hybrid acoustic ray tracer written in **C++17** and **CUDA**.

This engine simulates sound propagation in 3D spaces to generate realistic **Room Impulse Responses (RIR)**. It features a hybrid architecture that automatically detects your hardware:

* **NVIDIA GPU:** Runs massively parallel ray tracing using CUDA.
* **CPU (Mac/Linux):** Falls back to multi-threaded OpenMP + FFTW for compatibility.

## Features

* **Hybrid Core:** Seamlessly runs on MacBook Pro (CPU) or Cloud Servers (H100/A100 GPUs).
* **Room Shapes:** Shoebox, Dome (Hemisphere), and Arbitrary 3D Meshes (.obj).
* **Audio Processing:** Built-in Convolution engine to apply reverb to wav files instantly.

## Prerequisites

You only need **Docker** installed.

* **Docker Desktop** (Mac/Windows)
* **Docker Engine** (Linux)

## Quick Start

### 1. Build the Environment

The Docker setup handles all C++ toolchains (CMake, NVCC, FFTW) and Python helper dependencies.

```bash
docker compose up -d --build
```

### 2\. Enter the Container

All build and run commands must be executed inside the container.

```bash
docker compose exec dev bash
```

### 3\. Compile the Engine

Once inside the container, build the C++ project:

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

*Note: CMake will automatically detect if you have an NVIDIA GPU. If not, it will configure the project for CPU mode.*

## Usage

The executable is located at `./acoustic_sim` inside the `build/` directory.

### Basic Simulation (Shoebox)

Generate a 1-second Impulse Response for a 10m x 8m x 4m room.

```bash
./acoustic_sim --room shoebox --dims 10,8,4 --out warehouse.wav
```

### Dome Simulation

Simulate a hemispherical dome with a 15-meter radius.

```bash
./acoustic_sim --room dome --dims 15 --rays 500000 --out cathedral.wav
```

### Custom Mesh Simulation

You can use any `.obj` file. Place the file in your project root (e.g., `my_studio.obj`).

```bash
# Note: ../my_studio.obj because we are in the build/ folder
./acoustic_sim \
  --room mesh \
  --mesh ../my_studio.obj \
  --rays 100000 \
  --out studio_ir.wav
```

### Full Audio Processing (Reverb)

You can apply the room's acoustics to a sound file immediately using `--input` and `--mix`.

```bash
./acoustic_sim \
  --room shoebox \
  --dims 20,20,10 \
  --input ../dry_recording.wav \
  --mix 0.4 \
  --out wet_recording.wav
```

## Python Helpers

The container includes Python 3, `numpy`, and `trimesh` for generating test assets.

**Generate a Test Mesh (Icosphere)**
If you don't have a 3D model, run this one-liner to create a sphere mesh:

```bash
python3 -c "import trimesh; trimesh.creation.icosphere(radius=8).export('../test_sphere.obj')"
```

**Generate Test Audio**
If you need a "dry" sound to test reverb:

```bash
python3 ../generate_beat.py
```

## CLI Reference

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--room` | `shoebox`, `dome`, `mesh` | `shoebox` |
| `--dims` | Dimensions (`L,W,H` or `Radius`) | `10,5,3` |
| `--mesh` | Path to `.obj` file (Required for `mesh` mode) | - |
| `--rays` | Number of rays to trace (More = Higher Quality) | `100000` |
| `--input` | Path to input `.wav` for reverb processing | - |
| `--mix` | Wet/Dry mix (`0.0` to `1.0`) | `0.4` |
| `--out` | Output `.wav` filename | `out.wav` |

```
```