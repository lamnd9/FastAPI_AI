"""
Health Check Route
===================
Endpoint to verify the API service and model status.
"""

from fastapi import APIRouter

from utils.model_loader import get_model, get_class_labels

router = APIRouter()


@router.get("/health", summary="Kiểm tra trạng thái hệ thống")
async def health_check():
    """Check if the API service is alive and the model is loaded."""
    try:
        model = get_model()
        num_classes = len(get_class_labels())
        model_loaded = model is not None
    except RuntimeError:
        model_loaded = False
        num_classes = 0

    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "num_classes": num_classes,
        "message": "Cats vs Dogs Classifier API is running 🚀",
    }
