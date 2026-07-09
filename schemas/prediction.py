"""
Prediction Schemas
===================
Pydantic models for image classification prediction response.
"""

from pydantic import BaseModel, Field


class ClassPredictionItem(BaseModel):
    """A single prediction result class."""

    class_name: str = Field(
        ...,
        description="Tên lớp được dự đoán",
        examples=["dogs"],
    )
    probability: float = Field(
        ...,
        description="Xác suất dự đoán (0.0 - 1.0)",
        examples=[0.9854],
    )


class PredictResponse(BaseModel):
    """Response schema for prediction endpoint."""

    prediction: str = Field(
        ...,
        description="Nhãn lớp có xác suất cao nhất",
        examples=["dogs"],
    )
    confidence: float = Field(
        ...,
        description="Độ tin cậy của dự đoán chính (0.0 - 1.0)",
        examples=[0.9854],
    )
    top_5: list[ClassPredictionItem] = Field(
        ...,
        description="Danh sách các lớp có xác suất cao nhất",
    )
