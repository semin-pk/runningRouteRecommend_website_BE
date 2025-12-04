from fastapi import APIRouter

from app.api.v1 import recommend


api_router = APIRouter()

api_router.include_router(recommend.router, prefix="/api", tags=["recommend"])



