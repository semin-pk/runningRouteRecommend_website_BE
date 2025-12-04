from fastapi import FastAPI
from mangum import Mangum

from app.api.v1 import api_router
from app.core.security import create_limiter, setup_exception_handlers, setup_middlewares


def create_app() -> FastAPI:
    app = FastAPI(title="Running Route Recommender")

    limiter = create_limiter()
    setup_middlewares(app, limiter)
    setup_exception_handlers(app)

    app.include_router(api_router)

    return app


app = create_app()

handler = Mangum(app)



