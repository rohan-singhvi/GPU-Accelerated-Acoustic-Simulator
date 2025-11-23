FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

WORKDIR /app

# 1. Install System Dependencies
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# 2. Set up the environment
# Compile bytecode for faster startup
ENV UV_COMPILE_BYTECODE=1
# Explicitly add the .venv to PATH so you don't have to type "source ..."
ENV PATH="/app/.venv/bin:$PATH"

# 3. Install Python Dependencies
# Copy only the project files first (better caching)
COPY pyproject.toml uv.lock* /app/
ARG INSTALL_GPU=false
RUN if [ "$INSTALL_GPU" = "true" ]; then \
        echo "Installing GPU Dependencies (Base + CuPy)..." && \
        uv sync --extra gpu; \
    else \
        echo "Installing CPU Simulation Dependencies only..." && \
        uv sync; \
    fi

COPY . /app

# Default command
CMD ["bash"]
