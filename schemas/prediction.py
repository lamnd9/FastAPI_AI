"""
Prediction Schemas
===================
Pydantic models for prediction request and response validation.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Schema for prediction request body."""

    input_data: str = Field(
        ...,
        description="Input data for the AI model (text, base64-encoded image, etc.)",
        examples=["Hello, classify this text for me."],
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Optional model name to use for prediction",
        examples=["default_model"],
    )
    parameters: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional inference parameters (e.g., threshold, top_k)",
        examples=[{"threshold": 0.5, "top_k": 5}],
    )


class PredictionResponse(BaseModel):
    """Schema for prediction response body."""

    success: bool = Field(
        ...,
        description="Whether the prediction was successful",
    )
    message: str = Field(
        ...,
        description="Human-readable status message",
    )
    data: Optional[dict[str, Any]] = Field(
        default=None,
        description="Prediction results",
    )
