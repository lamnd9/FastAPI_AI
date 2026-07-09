"""
Model Loader Utility
=====================
Utility to load and cache AI/Deep Learning models.
"""

from pathlib import Path
from typing import Any, Optional

from utils.logger import logger
from config.settings import settings


class ModelLoader:
    """
    Singleton model loader that caches loaded models in memory.

    Usage:
        loader = ModelLoader()
        model = loader.load("my_model", "models/my_model.onnx")
        prediction = model.predict(input_data)
    """

    _instance: Optional["ModelLoader"] = None
    _models: dict[str, Any] = {}

    def __new__(cls) -> "ModelLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._models = {}
        return cls._instance

    def load(self, model_name: str, model_path: Optional[str] = None) -> Any:
        """
        Load a model by name. Returns cached version if already loaded.

        Args:
            model_name: Identifier for the model.
            model_path: Path to the model file. Defaults to settings.MODEL_PATH / model_name.

        Returns:
            The loaded model object.
        """
        if model_name in self._models:
            logger.info(f"📦 Model '{model_name}' loaded from cache.")
            return self._models[model_name]

        if model_path is None:
            model_path = str(Path(settings.MODEL_PATH) / model_name)

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                f"Please place your model weights in the '{settings.MODEL_PATH}' directory."
            )

        logger.info(f"⏳ Loading model '{model_name}' from {model_path}...")

        # ============================================================
        # TODO: Implement actual model loading based on framework
        # Examples:
        #
        # --- PyTorch ---
        # import torch
        # model = torch.load(model_path, map_location="cpu")
        # model.eval()
        #
        # --- ONNX Runtime ---
        # import onnxruntime as ort
        # model = ort.InferenceSession(model_path)
        #
        # --- TensorFlow / Keras ---
        # import tensorflow as tf
        # model = tf.keras.models.load_model(model_path)
        # ============================================================

        model = None  # Placeholder — replace with actual loading logic

        self._models[model_name] = model
        logger.info(f"✅ Model '{model_name}' loaded successfully.")

        return model

    def unload(self, model_name: str) -> None:
        """Remove a model from the cache."""
        if model_name in self._models:
            del self._models[model_name]
            logger.info(f"🗑️ Model '{model_name}' unloaded from cache.")

    def list_models(self) -> list[str]:
        """Return names of all currently loaded models."""
        return list(self._models.keys())


def get_model_loader() -> ModelLoader:
    """Get the singleton ModelLoader instance."""
    return ModelLoader()
