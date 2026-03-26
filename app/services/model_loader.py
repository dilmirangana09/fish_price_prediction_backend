import json
import joblib
import tensorflow as tf
from pathlib import Path
import keras

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "ml_models" / "deployed"

ANN_MODEL_PATH = MODEL_DIR / "ann_xgb_hybrid_ann_model.keras"
XGB_MODEL_PATH = MODEL_DIR / "ann_xgb_hybrid_xgb_model.pkl"
META_PATH = MODEL_DIR / "ann_xgb_hybrid_metadata.json"


def load_ann_xgb_hybrid():
    if not ANN_MODEL_PATH.exists():
        raise FileNotFoundError(f"ANN model not found: {ANN_MODEL_PATH}")

    if not XGB_MODEL_PATH.exists():
        raise FileNotFoundError(f"XGB model not found: {XGB_MODEL_PATH}")

    if not META_PATH.exists():
        raise FileNotFoundError(f"Metadata not found: {META_PATH}")

    ann_model = keras.models.load_model(ANN_MODEL_PATH, compile=False)
    xgb_model = joblib.load(XGB_MODEL_PATH)

    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return ann_model, xgb_model, metadata