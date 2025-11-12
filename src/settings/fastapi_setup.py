from fastapi import FastAPI
from .app_settings import Settings
from .limiter_setup import LimiterSetup
from .middleware_setup import MiddlewareSetup
from .exception_handler_setup import ExceptionHandlerSetup
from src.presentation.api.endpoints import health_check, recommend


class FastAPISetup:
    def __init__(self, settings: Settings, title: str = "Running Route Recommender"):
        self.app = FastAPI(title=title)
        self.settings = settings
        LimiterSetup(self.app)
        MiddlewareSetup(self.app, self.settings)
        ExceptionHandlerSetup(self.app, self.settings)
        self._setup_routers()

    def _setup_routers(self):
        self.app.include_router(health_check.router)
        self.app.include_router(recommend.router)

    def get_app(self) -> FastAPI:
        return self.app
