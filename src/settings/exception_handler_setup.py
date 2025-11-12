from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from src.settings.app_settings import Settings


class ExceptionHandlerSetup:
    def __init__(self, app: FastAPI, settings: Settings):
        self.app = app
        self.settings = settings
        self._setup()

    def _setup(self):
        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            return JSONResponse(
                status_code=exc.status_code,
                content={"error": exc.detail, "status_code": exc.status_code}
            )

        @self.app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            if self.settings.ENVIRONMENT == "production":
                return JSONResponse(
                    status_code=500,
                    content={"error": "Internal server error", "status_code": 500}
                )
            else:
                return JSONResponse(
                    status_code=500,
                    content={"error": str(exc), "status_code": 500}
                )
