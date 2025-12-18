import os
import cv2
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from ultralytics import YOLO

from constants import *

from vitpose.configs.ViTPose_base_coco_256x192 import model as model_cfg
from vitpose.models.model import ViTPose


def _require_file(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found at: {path}")


def load_models():
    """
    Load ALL models once at worker startup (RunPod-safe).

    - No GPU access at import time
    - Torch + TensorFlow imported lazily
    - TF memory growth enabled
    - DEVICE set here (NOT in constants.py)
    """
    global (
        ball_model,
        bat_model,
        yolo_model,
        vitpose,
        lstm_model,
        tokenizer,
        t5_model,
        DEVICE,
    )

    print("🔄 Loading models (fail-fast)...", flush=True)

    # ----------------------------
    # Lazy imports (CRITICAL)
    # ----------------------------
    import torch
    import tensorflow as tf

    # ----------------------------
    # Safe DEVICE selection
    # ----------------------------
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧠 Using device: {DEVICE}", flush=True)

    # ----------------------------
    # TensorFlow GPU memory safety
    # ----------------------------
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # ----------------------------
    # Validate required files
    # ----------------------------
    _require_file(BALL_MODEL_PATH, "BALL_MODEL (.pt)")
    _require_file(BAT_MODEL_PATH, "BAT_MODEL (.pt)")
    _require_file(PERSON_MODEL_PATH, "PERSON_MODEL yolov8n (.pt)")
    _require_file(VITPOSE_CKPT_PATH, "ViTPose checkpoint (.pth)")
    _require_file(LSTM_MODEL_PATH, "LSTM model (.keras)")
    _require_file(REF_CSV_PATH, "Reference CSV (1.csv)")
    _require_file(
        os.path.join(LLM_MODEL_DIR, "config.json"),
        "T5 model dir (config.json)",
    )

    # ----------------------------
    # YOLO models
    # ----------------------------
    ball_model = YOLO(BALL_MODEL_PATH)
    bat_model = YOLO(BAT_MODEL_PATH)
    yolo_model = YOLO(PERSON_MODEL_PATH)
    print("✅ YOLO models loaded", flush=True)

    # ----------------------------
    # ViTPose
    # ----------------------------
    ckpt = torch.load(
        VITPOSE_CKPT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
    state = ckpt.get("state_dict", ckpt)

    vitpose = ViTPose(model_cfg).to(DEVICE).eval()
    vitpose.load_state_dict(state, strict=False)
    print("✅ ViTPose loaded", flush=True)

    # ----------------------------
    # LSTM (TensorFlow)
    # ----------------------------
    lstm_model = tf.keras.models.load_model(
        LSTM_MODEL_PATH,
        compile=False,
    )
    print("✅ LSTM loaded", flush=True)

    # ----------------------------
    # T5 (offline, GPU)
    # ----------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_DIR,
        local_files_only=True,
    )
    t5_model = (
        AutoModelForSeq2SeqLM.from_pretrained(
            LLM_MODEL_DIR,
            local_files_only=True,
        )
        .to(DEVICE)
        .eval()
    )
    print("✅ T5 loaded", flush=True)

    print("✅ All models loaded successfully.", flush=True)


# =========================================================
# Utility functions
# =========================================================

def adaptive_square_crop(frame, target_size=CROP_SIZE):
    """Crop frame to square and resize to target_size."""
    h, w = frame.shape[:2]
    size = min(h, w)
    x1 = (w - size) // 2
    y1 = (h - size) // 2
    cropped = frame[y1 : y1 + size, x1 : x1 + size]
    return cv2.resize(cropped, (target_size, target_size))


def polygon_centroid(pts):
    a = np.array(pts, dtype=float)
    return float(a[:, 0].mean()), float(a[:, 1].mean())


def translate_polygon(pts, dx, dy):
    return [[int(round(x + dx)), int(round(y + dy))] for x, y in pts]
