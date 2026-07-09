"""
FastAPI Application Initialization
===================================
Main application factory for the Cats vs Dogs Classifier API.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from middleware.logging_middleware import LoggingMiddleware
from routes import health, prediction


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title=settings.APP_NAME,
        description="API nhận diện Chó và Mèo sử dụng mạng MobileNetV2 (Transfer Learning)",
        version=settings.APP_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # --- CORS Middleware ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Custom Middleware ---
    app.add_middleware(LoggingMiddleware)

    # --- Register Routes ---
    app.include_router(health.router, prefix="/api/v1", tags=["Health"])
    app.include_router(prediction.router, prefix="/api/v1", tags=["Prediction"])

    @app.on_event("startup")
    def startup_event():
        """Load AI model and class labels into memory on startup."""
        from utils.model_loader import load_model_and_classes

        print(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION} đang khởi động...")
        load_model_and_classes()
        print(f"✅ {settings.APP_NAME} sẵn sàng phục vụ!")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Actions to perform on application shutdown."""
        print(f"🛑 {settings.APP_NAME} đang tắt...")

    return app


app = create_app()
