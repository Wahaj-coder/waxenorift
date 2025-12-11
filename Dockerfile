# ============================================================
# BASE IMAGE — Python 3.10 slim (GPU-ready via CUDA wheels)
# ============================================================
FROM python:3.10-slim

# ------------------------------------------------------------
# Set working directory inside container
# ------------------------------------------------------------
WORKDIR /workspace

# ------------------------------------------------------------
# Install system-level dependencies
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    wget \
    gcc \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Create required folders
# ------------------------------------------------------------
RUN mkdir -p /workspace/models \
    /workspace/processed \
    /workspace/ViTPose_pytorch

# ------------------------------------------------------------
# Copy requirements first for layer caching
# (NOTE: this is the cleaned requirements.txt WITHOUT torch/tf)
# ------------------------------------------------------------
COPY requirements.txt .

# ------------------------------------------------------------
# Upgrade pip
# ------------------------------------------------------------
RUN pip install --no-cache-dir --upgrade pip

# ------------------------------------------------------------
# Install GPU-enabled PyTorch & TorchVision
# (These wheels include CUDA libs and will use GPU if drivers exist)
# Adjust versions if needed.
# ------------------------------------------------------------
RUN pip install --no-cache-dir \
    torch==2.4.1+cu121 \
    torchvision==0.19.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# ------------------------------------------------------------
# Install GPU-enabled TensorFlow
# TensorFlow 2.17+ supports "and-cuda" extra to pull CUDA-enabled build.
# If this errors due to versioning on your side, you can pin a different version.
# ------------------------------------------------------------
RUN pip install --no-cache-dir "tensorflow[and-cuda]==2.17.0"

# ------------------------------------------------------------
# Install the rest of your Python dependencies
# (No torch / tf here, they are already installed as GPU versions)
# ------------------------------------------------------------
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------
# Clone ViTPose repository and install its dependencies
# (This will reuse the already-installed torch where possible)
# ------------------------------------------------------------
RUN git clone https://github.com/jaehyunnn/ViTPose_pytorch.git /workspace/ViTPose_pytorch && \
    pip install --no-cache-dir -r /workspace/ViTPose_pytorch/requirements.txt

# ------------------------------------------------------------
# Copy app.py into container
# ------------------------------------------------------------
COPY app.py .

# ------------------------------------------------------------
# Expose API port
# ------------------------------------------------------------
EXPOSE 8000

# ------------------------------------------------------------
# Start Gunicorn with 2 workers (tune -w based on VRAM)
# ------------------------------------------------------------
CMD ["gunicorn", "-w", "2", "--threads", "2", "-b", "0.0.0.0:8000", "app:app"]
