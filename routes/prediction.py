"""
Prediction Route
==================
Endpoint for Cats vs Dogs classification using MobileNetV2.
Upload an image, get back the predicted animal type and confidence.
"""

import os

import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException

from schemas.prediction import PredictResponse, ClassPredictionItem
from utils.image_processing import preprocess_image, softmax
from utils.model_loader import get_model, get_class_labels
from utils.logger import logger

router = APIRouter()

# Supported image extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Dự đoán Chó hoặc Mèo từ ảnh gửi lên",
)
async def predict_image(file: UploadFile = File(...)):
    """
    Upload ảnh vật nuôi và nhận kết quả nhận diện (Chó hoặc Mèo).

    - Hỗ trợ định dạng: **JPG, JPEG, PNG, WebP**
    - Trả về: kết quả dự đoán cao nhất + xác suất chi tiết
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

        # 3. Preprocess image -> tensor (1, 160, 160, 3)
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

        # 7. Get top predictions (dynamically sized based on number of classes, up to 5)
        top_k = min(5, len(class_labels))
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        top_results = [
            ClassPredictionItem(
                class_name=class_labels[idx],
                probability=round(float(probabilities[idx]), 4),
            )
            for idx in top_indices
        ]

        logger.info(
            f"🔮 Kết quả: {predicted_class} ({confidence:.2%}) | "
            f"Chi tiết: {[r.class_name for r in top_results]}"
        )

        return PredictResponse(
            prediction=predicted_class,
            confidence=round(confidence, 4),
            top_5=top_results,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Lỗi xử lý ảnh: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi trong quá trình xử lý ảnh: {str(e)}",
        )
