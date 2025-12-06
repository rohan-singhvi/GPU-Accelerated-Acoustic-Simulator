# GPU-Accelerated Acoustic Simulator (C++/CUDA)

A high-performance, hybrid acoustic ray tracer written in **C++17** and **CUDA**.

This engine simulates sound propagation in 3D spaces to generate realistic **Room Impulse Responses (RIR)**. It features a **Stochastic Ray Tracing** engine that models physical phenomena like diffuse reflection (scattering), wall transmission (insulation), and absorption. It features a hybrid architecture that automatically detects your hardware:

* **NVIDIA GPU:** Runs massively parallel ray tracing using CUDA.
* **CPU (Mac/Linux):** Falls back to multi-threaded TBB + FFTW for compatibility.

## Features

* **Hybrid Core:** Seamlessly runs on MacBook Pro (CPU) or Cloud Servers (H100/A100 GPUs).
* **Material Physics:**
    * **Absorption:** Control energy loss per bounce (e.g., Concrete vs. Foam).
    * **Scattering:** Simulate surface roughness (Specular vs. Diffuse reflection).
    * **Transmission:** Stochastic modeling of sound passing through walls (Russian Roulette).
* **Room Shapes:** Shoebox, Dome (Hemisphere), and Arbitrary 3D Meshes (.obj).
* **Analysis Tools:** Built-in Python scripts for audio generation and spectral visualization.

## Prerequisites

You only need **Docker** installed.

* **Docker Desktop** (Mac/Windows)
* **Docker Engine** (Linux)

## Quick Start

### 1. Build the Environment

The Docker setup handles all C++ toolchains (CMake, NVCC, FFTW) and Python analysis dependencies (NumPy, Matplotlib, SoundFile, Scipy, Trimesh).

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

### Material Physics Examples

You can tweak the material properties to simulate different environments.

**1. The "Cathedral" (Reflective, Rough/Diffuse)**
Low absorption, high scattering.

```bash
./acoustic_sim \
  --room shoebox --dims 20,15,10 \
  --absorption 0.05 \
  --scattering 0.7 \
  --out cathedral.wav
```

**2. The "Paper House" (High Transmission)**
Sound leaks through thin walls (Transmission \> 0).

```bash
./acoustic_sim \
  --room shoebox --dims 5,5,3 \
  --trans 0.5 --thick 0.05 \
  --out thin_walls.wav
```

### Custom Mesh Simulation

You can use any `.obj` file. Place the file in your project root.

```bash
# Note: ../my_studio.obj because we are in the build/ folder
./acoustic_sim \
  --room mesh \
  --mesh ../my_studio.obj \
  --rays 100000 \
  --absorption 0.2 \
  --out studio_ir.wav
```

### Full Audio Processing (Reverb)

Apply the room's acoustics to a sound file immediately using `--input` and `--mix`.

```bash
./acoustic_sim \
  --room shoebox --dims 12,12,6 \
  --absorption 0.1 --scattering 0.5 \
  --input ../dry_recording.wav \
  --mix 0.4 \
  --out wet_result.wav
```

## Python Analysis Tools

The container includes scripts to generate test audio and visualize the physics.

**1. Generate Test Audio**
Creates a sparse techno beat (`techno_dry.wav`) designed for testing reverb tails.

```bash
python3 ../generate_beat.py
```

**2. Visualize Acoustics**
Generates a 3-panel analysis plot (Waveform, Spectrogram, Energy Decay) to verify your physics settings.

```bash
# Compare the dry input against your simulation output
python3 ../visualize.py techno_dry.wav wet_result.wav
```

*Output: `acoustic_analysis.png`*

**3. Generate a Test Mesh (Icosphere)**
If you don't have a 3D model, run this one-liner to create a sphere mesh:
```bash
python3 -c "import trimesh; trimesh.creation.icosphere(radius=8).export('../test_sphere.obj')"
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
| `--absorption` | Wall energy loss (`0.0`=Mirror, `1.0`=Dead) | `0.1` |
| `--scattering` | Surface roughness (`0.0`=Smooth, `1.0`=Diffuse) | `0.1` |
| `--trans` | Transmission probability (`0.0`=Opaque, `1.0`=Clear) | `0.0` |
| `--thick` | Wall thickness (meters) for transmission calculation | `0.2` |

