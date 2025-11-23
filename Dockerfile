# ==========================================
# Stage 1: Builder (Compiles specific libs)
# ==========================================
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install build tools and Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    git \
    build-essential \
    curl \
    ca-certificates \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       python3.11 \
       python3.11-dev \
       python3.11-venv \
       python3.11-distutils

# Create a virtual environment to isolate dependencies
RUN python3.11 -m venv /opt/venv
# Add venv to PATH so all subsequent pip installs go here automatically
ENV PATH="/opt/venv/bin:$PATH"

# Install Pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# 1. Install PyTorch (GPU) first
# We use the specific index URL to ensure we get the CUDA version, not CPU
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 2. Install Requirements
COPY requirements.txt .
# Remove 'appnope' (macOS specific) before installing to prevent errors
RUN sed -i '/appnope/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

# 3. Install Heavy Git Libraries (Detectron2, MoGe)
# Detectron2 requires the nvcc compiler present in this stage
RUN pip install --no-cache-dir \
    "git+https://github.com/facebookresearch/detectron2.git@a1ce2f9" \
    --no-build-isolation

RUN pip install --no-cache-dir "git+https://github.com/microsoft/MoGe.git"


# ==========================================
# Stage 2: Runtime (Slimmer final image)
# ==========================================
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Force EGL for PyRender (headless rendering)
ENV PYOPENGL_PLATFORM=egl
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install Python 3.11 and Runtime Libraries (GL, EGL, OpenCV deps)
# We DO NOT need gcc, git, or build-essential here
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       python3.11 \
       python3.11-venv \
       libegl1 \
       libgl1 \
       libgl1-mesa-dri \
       libglib2.0-0 \
       mesa-utils \
       ffmpeg \
       libsm6 \
       libxext6 \
       libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-built virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY . .

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app
USER appuser

CMD ["python", "main.py"]