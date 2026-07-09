"""
Logging Middleware
==================
Middleware to log request/response details for monitoring and debugging.
"""

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from utils.logger import logger


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs incoming requests and outgoing responses."""

    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.time()

        # Log incoming request
        logger.info(
            f"📥 Request: {request.method} {request.url.path} "
            f"| Client: {request.client.host if request.client else 'unknown'}"
        )

        try:
            response: Response = await call_next(request)
        except Exception as e:
            logger.error(f"❌ Unhandled exception: {e}")
            raise

        # Calculate processing time
        process_time = round((time.time() - start_time) * 1000, 2)

        # Log outgoing response
        logger.info(
            f"📤 Response: {request.method} {request.url.path} "
            f"| Status: {response.status_code} "
            f"| Duration: {process_time}ms"
        )

        # Add processing time header
        response.headers["X-Process-Time-Ms"] = str(process_time)

        return response
