"""
Health Check Route
===================
Endpoint to verify the API service is running.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health", summary="Health Check")
async def health_check():
    """Check if the API service is alive and healthy."""
    return {
        "status": "healthy",
        "message": "FastAPI AI Service is running 🚀",
    }
