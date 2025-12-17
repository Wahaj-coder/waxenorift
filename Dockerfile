FROM python:3.10-slim

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    wget \
    gcc \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    unzip \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# gdown for Google Drive downloads
RUN pip install --no-cache-dir gdown

# Download ALL model weights (OFFLINE, FAIL FAST)
RUN set -eux; \
    mkdir -p /workspace/models; \
    \
    echo "Downloading cricket_ball_detector.pt"; \
    gdown --id 1RFR7QNG0KS8u68IiB4ZR4fZAvyRwxyZ7 \
        -O /workspace/models/cricket_ball_detector.pt; \
    \
    echo "Downloading bestBat.pt"; \
    gdown --id 1MQR-tOl86pAWfhtUtg7PDDDmsTq0eUM1 \
        -O /workspace/models/bestBat.pt; \
    \
    echo "Downloading vitpose-b-multi-coco.pth"; \
    gdown --id 1mHoFS6PEGGx3E0INBdSfFyUr5kUtOUNs \
        -O /workspace/models/vitpose-b-multi-coco.pth; \
    \
    echo "Downloading thirdlstm.weights.h5"; \
    gdown --id 12ZJHVQ4ew5WsNlyo6iTDTH4AH1cN4pLX \
        -O /workspace/models/thirdlstm.weights.h5; \
    \
    echo "Downloading 1.csv"; \
    gdown --id 1aKrG286A-JQecHA2IhIuR03fVxd-yMsx \
        -O /workspace/models/1.csv; \
    \
    echo "Downloading cricket_t5_final_clean.zip"; \
    gdown --id 1XheZOO2UO4ZVtupBSNXQwaT09-S-WWtB \
        -O /workspace/models/cricket_t5_final_clean.zip; \
    \
    echo "Unzipping cricket_t5_final_clean.zip"; \
    unzip /workspace/models/cricket_t5_final_clean.zip \
        -d /workspace/models/cricket_t5_final_clean; \
    rm /workspace/models/cricket_t5_final_clean.zip; \
    \
    echo "Downloading YOLOv8n.pt"; \
    gdown --id 19pOyZ3K7zKXUaTAE2TFFmf5Ze9eqnfbc \
        -O /workspace/models/yolov8n.pt; \
    \
    echo "All models downloaded successfully"

# Copy requirements.txt
COPY requirements.txt .

# Ensure pip & setuptools are up-to-date
RUN python -m ensurepip --upgrade
RUN python -m pip install --upgrade setuptools

# Install Python dependencies
RUN python -m pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY src/ ./src/

# Set entrypoint
CMD ["python", "src/handler.py"]

