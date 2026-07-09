"""
Prediction Schemas
===================
Pydantic models for food recognition prediction request and response.
"""

from pydantic import BaseModel, Field


class FoodPredictionItem(BaseModel):
    """A single food prediction result."""

    class_name: str = Field(
        ...,
        description="Tên món ăn được dự đoán",
        examples=["Pho"],
    )
    probability: float = Field(
        ...,
        description="Xác suất dự đoán (0.0 - 1.0)",
        examples=[0.9523],
    )


class PredictResponse(BaseModel):
    """Response schema for food prediction endpoint."""

    prediction: str = Field(
        ...,
        description="Tên món ăn có xác suất cao nhất",
        examples=["Pho"],
    )
    confidence: float = Field(
        ...,
        description="Độ tin cậy của dự đoán chính (0.0 - 1.0)",
        examples=[0.9523],
    )
    top_5: list[FoodPredictionItem] = Field(
        ...,
        description="Top 5 món ăn có xác suất cao nhất",
    )
