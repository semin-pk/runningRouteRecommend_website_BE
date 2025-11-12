from typing import List
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from src.settings.app_settings import Settings


class MiddlewareSetup:
    def __init__(self, app: FastAPI, settings: Settings):
        self.app = app
        self.settings = settings
        self._setup()

    def _setup(self):
        if self.settings.ENVIRONMENT == "production":
            self.app.add_middleware(
                TrustedHostMiddleware,
                allowed_hosts=[
                    host for host in self.settings.ALLOWED_HOSTS
                    if host not in ["localhost", "127.0.0.1", "0.0.0.0"]
                ]
            )
        else:
            self.app.add_middleware(
                TrustedHostMiddleware,
                allowed_hosts=self.settings.ALLOWED_HOSTS
            )

        allowed_origins = self.settings.ALLOWED_ORIGINS
        if self.settings.ENVIRONMENT == "production":
            allowed_origins = [
                origin for origin in self.settings.ALLOWED_ORIGINS
                if not origin.startswith("http://localhost") and not origin.startswith("http://127.0.0.1")
            ]

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )

        @self.app.middleware("http")
        async def add_security_headers(request: Request, call_next):
            response = await call_next(request)
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            return response
