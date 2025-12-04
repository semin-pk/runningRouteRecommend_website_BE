from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.core.config import ENVIRONMENT, get_allowed_origins


def create_limiter() -> Limiter:
    return Limiter(key_func=get_remote_address)


def setup_middlewares(app: FastAPI, limiter: Limiter) -> None:
    """Register common middlewares (CORS, trusted host, security headers, rate limiter handlers)."""
    # rate limiter
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # trusted hosts
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=[
            "www.run2yourstyle.com",
            "run2yourstyle.com",
            "*.amazonaws.com",
            "localhost",
        ],
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=get_allowed_origins(),
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    # security headers
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers[
            "Strict-Transport-Security"
        ] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response


def setup_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers."""

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.detail, "status_code": exc.status_code},
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        if ENVIRONMENT == "production":
            return JSONResponse(
                status_code=500,
                content={"error": "Internal server error", "status_code": 500},
            )
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "status_code": 500},
        )



