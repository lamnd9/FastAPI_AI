"""
FastAPI Application Initialization
===================================
Main application factory for the FastAPI AI service.
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
        description="FastAPI base structure for AI model serving",
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
    async def startup_event():
        """Actions to perform on application startup."""
        print(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION} is starting up...")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Actions to perform on application shutdown."""
        print(f"🛑 {settings.APP_NAME} is shutting down...")

    return app


app = create_app()
