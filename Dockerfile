FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ----------------------------
# Environment
# ----------------------------
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

WORKDIR /workspace

# ----------------------------
# System dependencies
# ----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    git \
    wget \
    unzip \
    libgl1 \
    libglib2.0-0 \
    libgeos-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Make python available as `python`
RUN ln -sf /usr/bin/python3 /usr/bin/python

# ----------------------------
# Upgrade pip tooling
# ----------------------------
RUN python -m pip install --upgrade pip setuptools wheel

# ----------------------------
# gdown for offline models
# ----------------------------
RUN pip install --no-cache-dir gdown

# ----------------------------
# Download ALL model weights (baked into image)
# ----------------------------
RUN set -eux; \
    mkdir -p /workspace/models; \
    \
    gdown --id 1RFR7QNG0KS8u68IiB4ZR4fZAvyRwxyZ7 -O /workspace/models/cricket_ball_detector.pt; \
    gdown --id 1MQR-tOl86pAWfhtUtg7PDDDmsTq0eUM1 -O /workspace/models/bestBat.pt; \
    gdown --id 1mHoFS6PEGGx3E0INBdSfFyUr5kUtOUNs -O /workspace/models/vitpose-b-multi-coco.pth; \
    gdown --id 1G_tJzRtSKaTJmoet0Cma8dCjgJCifTMu -O /workspace/models/thirdlstm_shot_classifierupdated.keras; \
    gdown --id 1aKrG286A-JQecHA2IhIuR03fVxd-yMsx -O /workspace/models/1.csv; \
    gdown --id 1XheZOO2UO4ZVtupBSNXQwaT09-S-WWtB -O /workspace/models/cricket_t5_final_clean.zip; \
    unzip /workspace/models/cricket_t5_final_clean.zip -d /workspace/models/cricket_t5_final_clean; \
    rm /workspace/models/cricket_t5_final_clean.zip; \
    gdown --id 19pOyZ3K7zKXUaTAE2TFFmf5Ze9eqnfbc -O /workspace/models/yolov8n.pt

# ----------------------------
# PyTorch (CUDA 12.1)
# ----------------------------
RUN pip install --no-cache-dir \
    torch==2.4.1+cu121 \
    torchvision==0.19.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# ----------------------------
# TensorFlow (stable CUDA-compatible)
# ----------------------------
RUN pip install --no-cache-dir tensorflow==2.15.1

# ----------------------------
# Remaining Python dependencies
# (torch / torchvision MUST NOT be here)
# ----------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------
# App code
# ----------------------------
COPY src/ ./src/

# ----------------------------
# Entry point
# ----------------------------
CMD ["python", "src/handler.py"]
