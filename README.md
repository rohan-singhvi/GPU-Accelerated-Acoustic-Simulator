# GPU-Accelerated Acoustic Simulator

A high-performance room acoustic simulation engine using Python, CUDA (via Numba), and `uv` for dependency management. This project simulates sound propagation in 3D spaces (Shoebox or Dome topologies for now) to generate realistic Room Impulse Responses (RIR) and applies them to audio files.

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
2.  **Process:** Apply that impulse response to a "dry" audio file (like a voice recording) to simulate reverb.

### Step 1: Acoustic Simulation (Generate IR)

Run the simulator to ray-trace the room acoustics.

```bash
# Basic usage (Shoebox room)
python acoustic_simulator.py --out room_impulse.wav

python acoustic_simulator.py --room shoebox --dims "10,5,3" --out warehouse.wav

# Advanced usage (Dome room with specific dimensions)
# --dims for Dome is the radius in meters
python acoustic_simulator.py --room dome --dims 15 --rays 500000 --out cathedral.wav
python acoustic_simulator.py --room dome --dims "10" --source "0,5,0" --listener "3,1,0" --out dome.wav
```

**Arguments:**

  * `--room`: Room shape (`shoebox` or `dome`).
  * `--dims`: Dimensions. For Shoebox: `L,W,H` (e.g., `10,5,3`). For Dome: `Radius` (e.g., `15`).
  * `--rays`: Number of rays to trace (more = higher quality, slower).
  * `--source`: Source coordinates `x,y,z`.
  * `--listener`: Microphone coordinates `x,y,z`.

### Step 2: Audio Processing (Apply Reverb)

To hear the result, you need a "dry" audio file (e.g., `my_voice.wav`) inside your project folder.

```bash
# Apply the generated impulse to your audio
python process_audio.py my_voice.wav --ir room_impulse.wav --mix 0.4
```

**Arguments:**

  * `input_file`: The wav/flac file you want to process.
  * `--ir`: The impulse response generated in Step 1.
  * `--mix`: Wet/Dry mix (0.0 = Original sound, 1.0 = Full Reverb).

The output will be saved as `processed_output.wav`.

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

If the above does not work, you may need to delete the .venv, i.e.
```bash
docker compose down -v

rm -rf .venv

docker compose up --build -d
```


### CUDA vs. Simulation Mode

The container is configured to run in **Simulation Mode** (CPU) by default for compatibility with non-NVIDIA hardware.

  * **Simulation Mode:** Uses `NUMBA_ENABLE_CUDASIM=1`, and `INSTALL_GPU="false"`. Slower, but works everywhere.
  * **Real GPU Mode:** To use a real NVIDIA GPU, edit `docker-compose.yml`, set `NUMBA_ENABLE_CUDASIM=0` + `INSTALL_GPU: "true"` (under build.args), and rebuild the container. 
  Ensure the NVIDIA Container Toolkit is installed on your host.
