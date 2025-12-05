from fastapi import APIRouter

from app.api.v1 import recommend, store


api_router = APIRouter()

api_router.include_router(recommend.router, prefix="/api", tags=["recommend"])
api_router.include_router(store.router, prefix="/api", tags=["store"])



