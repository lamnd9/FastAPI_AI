"""
Image Processing Utility
==========================
Preprocessing functions for the EfficientNetV2 food recognition model.
"""

import io

import numpy as np
from PIL import Image

from config.settings import settings


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess raw image bytes into a model-ready tensor.

    Steps:
        1. Open image and convert to RGB
        2. Resize to 224x224
        3. Convert to float32 numpy array
        4. Normalize pixel values to [0.0, 1.0]
        5. Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)

    Args:
        image_bytes: Raw bytes of the uploaded image file.

    Returns:
        Numpy array with shape (1, 224, 224, 3) and dtype float32.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_resized = img.resize((settings.IMAGE_SIZE, settings.IMAGE_SIZE))

    x = np.array(img_resized, dtype=np.float32)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)

    return x


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Apply softmax to convert raw logits into probability distribution.

    Uses the numerical stability trick of subtracting max value
    to prevent overflow.

    Args:
        logits: 1D array of raw model output logits.

    Returns:
        1D array of probabilities summing to 1.0.
    """
    exp_x = np.exp(logits - np.max(logits))
    return exp_x / np.sum(exp_x)
