import os

import cv2
import numpy as np
import tensorflow as tf
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from ultralytics import YOLO

from constants import *

from vitpose.configs.ViTPose_base_coco_256x192 import model as model_cfg
from vitpose.models.model import ViTPose


# =========================================================
# UTILS
# =========================================================

def _require_file(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found at: {path}")


# =========================================================
# MODEL LOADER (FAIL FAST)
# =========================================================

def load_models():
    """
    Load ALL models once at worker startup.

    - TF 2.12 safe
    - No batch_shape usage
    - LSTM weights-only loading
    - Exact architecture match to training model
    """

    global ball_model, bat_model, yolo_model
    global vitpose, lstm_model, tokenizer, t5_model

    print("🔄 Loading models (fail-fast)...")

    # -----------------------------------------------------
    # Verify required files
    # -----------------------------------------------------
    _require_file(BALL_MODEL_PATH, "BALL_MODEL (.pt)")
    _require_file(BAT_MODEL_PATH, "BAT_MODEL (.pt)")
    _require_file(PERSON_MODEL_PATH, "PERSON_MODEL yolov8n (.pt)")
    _require_file(VITPOSE_CKPT_PATH, "ViTPose checkpoint (.pth)")
    _require_file(LSTM_MODEL_PATH, "LSTM weights (.weights.h5)")
    _require_file(REF_CSV_PATH, "Reference CSV (1.csv)")
    _require_file(
        os.path.join(LLM_MODEL_DIR, "config.json"),
        "T5 model dir (config.json)"
    )

    # -----------------------------------------------------
    # YOLO models
    # -----------------------------------------------------
    ball_model = YOLO(BALL_MODEL_PATH)
    bat_model = YOLO(BAT_MODEL_PATH)
    yolo_model = YOLO(PERSON_MODEL_PATH)
    print("✅ YOLO models loaded")

    # -----------------------------------------------------
    # ViTPose
    # -----------------------------------------------------
    ckpt = torch.load(VITPOSE_CKPT_PATH, map_location=DEVICE)
    state = ckpt.get("state_dict", ckpt)

    vp = ViTPose(model_cfg).to(DEVICE).eval()
    vp.load_state_dict(state, strict=False)

    vitpose = vp
    print("✅ ViTPose loaded")

    # -----------------------------------------------------
    # LSTM (EXACT MATCH to training summary)
    # -----------------------------------------------------
    lstm_model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(
                shape=(WINDOW_SIZE, LSTM_EXPECTED_FEATURES),
                name="input_layer"
            ),

            # lstm_4
            tf.keras.layers.LSTM(
                256,
                return_sequences=True,
                name="lstm_4"
            ),
            tf.keras.layers.BatchNormalization(name="batch_normalization_4"),
            tf.keras.layers.Dropout(0.2, name="dropout_6"),

            # lstm_5
            tf.keras.layers.LSTM(
                128,
                return_sequences=False,
                name="lstm_5"
            ),
            tf.keras.layers.BatchNormalization(name="batch_normalization_5"),
            tf.keras.layers.Dropout(0.2, name="dropout_7"),

            # Dense layers
            tf.keras.layers.Dense(
                64,
                activation="relu",
                name="dense_4"
            ),
            tf.keras.layers.Dropout(0.2, name="dropout_8"),

            # IMPORTANT: final Dense has NO activation (linear)
            tf.keras.layers.Dense(
                len(LABEL_CLASSES),
                name="dense_5"
            ),
        ]
    )

    # Explicit build (critical for weights loading)
    lstm_model.build(
        input_shape=(None, WINDOW_SIZE, LSTM_EXPECTED_FEATURES)
    )

    # Load weights (TF 2.12 compatible)
    lstm_model.load_weights(LSTM_MODEL_PATH)

    print("✅ LSTM weights loaded")

    # -----------------------------------------------------
    # T5 (offline)
    # -----------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_DIR,
        local_files_only=True
    )

    t5_model = (
        AutoModelForSeq2SeqLM
        .from_pretrained(LLM_MODEL_DIR, local_files_only=True)
        .to(DEVICE)
        .eval()
    )

    print("✅ T5 loaded")
    print("✅ Model loading done.")


# =========================================================
# HELPERS
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
