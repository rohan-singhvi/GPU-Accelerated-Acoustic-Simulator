# GPU-Accelerated Acoustic Simulator

A high-performance room acoustic simulation engine using Python, CUDA (via Numba), and `uv` for dependency management. This project simulates sound propagation in 3D spaces—supporting Shoeboxes, Domes, and Arbitrary Meshes—to generate realistic Room Impulse Responses (RIR) and applies them to audio files.

## Prerequisites

You do **not** need Python, CUDA, or `uv` installed on your local machine. You only need:

  * **Docker Desktop** (Mac/Windows) or **Docker Engine** (Linux)
  * **Git**

## Quick Start & Installation

This project uses a specific Docker setup to handle dependencies.

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/rohan-singhvi/GPU-Accelerated-Acoustic-Simulator
    cd GPU-Accelerated-Acoustic-Simulator
    ```

2.  **Build and Start the Container:**
    This command builds the image and sets up the internal virtual environment.

    ```bash
    docker compose up --build -d
    ```

3.  **Enter the Development Container:**
    All simulation commands must be run **inside** the container.

    ```bash
    docker compose exec cuda-dev bash
    ```

### Verification

Once inside the container, verify that the environment is using the correct virtual environment (managed by `uv`).

Run this command inside the container:

```bash
which python
```

**Success:** Output should be `/app/.venv/bin/python`
**Failure:** If output is `/usr/local/bin/python`, the volume mount has failed. See *Troubleshooting* below.

-----

## Usage

The workflow consists of two steps:

1.  **Simulate:** Generate a Room Impulse Response (`.wav` file) based on room geometry.
2.  **Process:** Apply that impulse response to a "dry" audio file to simulate reverb.

### Step 1: Acoustic Simulation (Generate IR)

Run the simulator to ray-trace the room acoustics.

**Basic Shapes (Shoebox & Dome)**

```bash
# Basic usage (Shoebox room)
python acoustic_simulator.py --room shoebox --dims "10,5,3" --out warehouse.wav

# Dome room (Dimensions = Radius)
python acoustic_simulator.py --room dome --dims 15 --rays 500000 --out cathedral.wav
```

**Arbitrary Mesh (.obj/.stl)**

If you don't have a 3D model handy, you can generate a test sphere inside the container:

```bash
# Generate a test object (Icosphere)
python -c "import trimesh; trimesh.creation.icosphere(radius=8).export('test_room.obj')"
```

Then run the simulator against the mesh:

```bash
python acoustic_simulator.py \
  --room mesh \
  --mesh-file test_room.obj \
  --rays 10000 \
  --out mesh_reverb.wav
```

**Arguments:**

  * `--room`: Room shape (`shoebox`, `dome`, or `mesh`).
  * `--dims`: Dimensions. For Shoebox: `L,W,H` (e.g., `10,5,3`). For Dome: `Radius` (e.g., `15`).
  * `--mesh-file`: Path to .obj/.stl file (Required if room is `mesh`).
  * `--rays`: Number of rays to trace (more = higher quality, slower).
  * `--source`: Source coordinates `x,y,z`.
  * `--listener`: Microphone coordinates `x,y,z`.

### Step 2: Audio Processing (Apply Reverb)

To hear the result, you need a "dry" audio file inside your project folder. You can generate a synthetic techno beat for testing:

```bash
# Generate test audio
python generate_beat.py
```

Then apply the impulse response:

```bash
# Apply the generated impulse to your audio
python process_audio.py techno_dry.wav --ir mesh_reverb.wav --mix 0.4 --output final_result
```

**Arguments:**

  * `input_file`: The wav/flac file you want to process.
  * `--ir`: The impulse response generated in Step 1.
  * `--mix`: Wet/Dry mix (0.0 = Original sound, 1.0 = Full Reverb).
  * `--output`: Filename for the result (saved as `.wav`).

-----

## Architecture & Troubleshooting

### Dependency Management (uv)

This project uses **uv** for extremely fast Python package management.

  * Dependencies are defined in `pyproject.toml`.
  * The `Dockerfile` handles `uv sync` automatically.

### Troubleshooting: "Module Not Found"

If you see `ModuleNotFoundError: No module named 'numpy'`, your Docker volume has likely de-synced the internal virtual environment from the host mount.

To fix this, perform a **Volume Reset**:

1.  Exit the container.
2.  Run the following commands on your host machine to destroy the stale volumes and rebuild:

<!-- end list -->

```bash
# Stop containers and delete volumes (Crucial Step)
docker compose down -v

# Rebuild and start fresh
docker compose up --build -d
```

### CUDA vs. Simulation Mode

The container is configured to run in **Simulation Mode** (CPU) by default for compatibility with non-NVIDIA hardware (like MacBook Pro).

  * **Simulation Mode (Mac/CPU):**

      * Uses `NUMBA_ENABLE_CUDASIM=1`.
      * Build Arg: `INSTALL_GPU="false"`.
      * **Result:** Runs slowly on CPU, but allows logic development without an NVIDIA card.

  * **Real GPU Mode (Linux/Cloud):**
    To use a real NVIDIA GPU, you must enable the GPU libraries and disable the simulator:

    1.  Edit `docker-compose.yml`:
          * Set `INSTALL_GPU: "true"` (under `build.args`).
          * Set `NUMBA_ENABLE_CUDASIM=0` (under `environment`).
    2.  **Rebuild the container:**
        ```bash
        docker compose up -d --build
        ```

    <!-- end list -->

      * **Result:** Installs `cupy-cuda12x` and runs at full hardware speed. Ensure the NVIDIA Container Toolkit is installed on your host.