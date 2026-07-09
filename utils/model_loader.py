"""
Model Loader Utility
=====================
Load and cache the EfficientNetV2 food recognition model and class labels.
"""

import os
from pathlib import Path
from typing import Optional

import tensorflow as tf
from tensorflow.keras import models

from config.settings import settings
from utils.logger import logger

# --- Singleton model and class labels ---
_model: Optional[tf.keras.Model] = None
_class_labels: list[str] = []


def load_model_and_classes() -> None:
    """
    Load the EfficientNetV2 model and class labels into memory.
    Called once during application startup.
    """
    global _model, _class_labels

    base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
    model_path = base_dir / settings.MODEL_PATH / settings.MODEL_FILE
    classes_path = base_dir / settings.CLASSES_PATH

    # 1. Load class labels
    if not classes_path.exists():
        raise RuntimeError(
            f"Không tìm thấy file nhãn tại {classes_path}. "
            f"Vui lòng đặt file product_classes.txt vào thư mục gốc dự án."
        )

    with open(classes_path, "r", encoding="utf-8") as f:
        _class_labels = [line.strip() for line in f.readlines() if line.strip()]

    logger.info(f"📋 Đã tải {len(_class_labels)} nhãn món ăn từ {classes_path.name}")

    # 2. Load Keras model
    if not model_path.exists():
        raise RuntimeError(
            f"Không tìm thấy file model tại {model_path}. "
            f"Vui lòng đặt model weights vào thư mục '{settings.MODEL_PATH}'."
        )

    logger.info(f"⏳ Đang nạp model EfficientNetV2 từ {model_path.name}...")
    _model = models.load_model(str(model_path))
    logger.info("✅ Nạp model EfficientNetV2 thành công!")


def get_model() -> tf.keras.Model:
    """Get the loaded Keras model. Raises if not loaded yet."""
    if _model is None:
        raise RuntimeError("Model chưa được nạp. Hãy gọi load_model_and_classes() trước.")
    return _model


def get_class_labels() -> list[str]:
    """Get the loaded class labels list. Raises if not loaded yet."""
    if not _class_labels:
        raise RuntimeError("Class labels chưa được nạp. Hãy gọi load_model_and_classes() trước.")
    return _class_labels
