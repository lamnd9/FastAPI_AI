"""
Prediction Route
==================
Endpoint for food recognition using EfficientNetV2 model.
Upload an image, get back the predicted dish name and top-5 results.
"""

import os

import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException

from schemas.prediction import PredictResponse, FoodPredictionItem
from utils.image_processing import preprocess_image, softmax
from utils.model_loader import get_model, get_class_labels
from utils.logger import logger

router = APIRouter()

# Supported image extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Dự đoán món ăn từ ảnh gửi lên",
)
async def predict_food(file: UploadFile = File(...)):
    """
    Upload ảnh món ăn và nhận kết quả nhận diện.

    - Hỗ trợ định dạng: **JPG, JPEG, PNG, WebP**
    - Trả về: món ăn dự đoán cao nhất + top 5 kết quả
    """

    # 1. Validate file extension
    extension = os.path.splitext(file.filename or "")[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Chỉ hỗ trợ ảnh định dạng {', '.join(ALLOWED_EXTENSIONS)}. "
                   f"File gửi lên có định dạng: '{extension}'",
        )

    try:
        # 2. Read image bytes
        image_bytes = await file.read()
        logger.info(
            f"🖼️ Nhận ảnh: {file.filename} ({len(image_bytes):,} bytes)"
        )

        # 3. Preprocess image -> tensor (1, 224, 224, 3)
        input_tensor = preprocess_image(image_bytes)

        # 4. Run model inference
        model = get_model()
        preds = model.predict(input_tensor)
        raw_logits = preds[0]

        # 5. Convert logits -> probabilities via softmax
        probabilities = softmax(raw_logits)

        # 6. Get top-1 prediction
        class_labels = get_class_labels()
        predicted_idx = int(np.argmax(probabilities))
        predicted_class = class_labels[predicted_idx]
        confidence = float(probabilities[predicted_idx])

        # 7. Get top-5 predictions
        top_5_indices = np.argsort(probabilities)[-5:][::-1]
        top_5_results = [
            FoodPredictionItem(
                class_name=class_labels[idx],
                probability=round(float(probabilities[idx]), 4),
            )
            for idx in top_5_indices
        ]

        logger.info(
            f"🔮 Kết quả: {predicted_class} ({confidence:.2%}) | "
            f"Top-5: {[r.class_name for r in top_5_results]}"
        )

        return PredictResponse(
            prediction=predicted_class,
            confidence=round(confidence, 4),
            top_5=top_5_results,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Lỗi xử lý ảnh: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi trong quá trình xử lý ảnh: {str(e)}",
        )
