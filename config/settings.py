"""
Application Settings
=====================
Central configuration using pydantic-settings.
Values are loaded from environment variables or .env file.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- App ---
    APP_NAME: str = "FastAPI AI Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # --- Server ---
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1

    # --- CORS ---
    ALLOWED_ORIGINS: list[str] = ["*"]

    # --- Model ---
    MODEL_PATH: str = "models/"
    MODEL_NAME: str = "default_model"

    # --- Logging ---
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "logs/"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
