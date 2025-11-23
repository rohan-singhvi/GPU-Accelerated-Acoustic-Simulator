FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

WORKDIR /app

# sys deps
RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# environment
# compile bytecode for faster startup
ENV UV_COMPILE_BYTECODE=1
ENV PATH="/app/.venv/bin:$PATH"

# python deps
# copy only the project files first (better caching)
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

CMD ["bash"]
