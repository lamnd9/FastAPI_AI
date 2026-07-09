"""
Server Entry Point
===================
Python file containing codes to host the API service.
Run with: python server.py
"""

import uvicorn

from config.settings import settings

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
        workers=settings.WORKERS,
    )
