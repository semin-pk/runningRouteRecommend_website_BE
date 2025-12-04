from typing import Dict

from fastapi import APIRouter, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.schemas.recommend import RecommendRequest, RecommendResponse
from app.services.recommend_service import recommend_route


router = APIRouter()

limiter: Limiter = Limiter(key_func=get_remote_address)


@router.get("/health_check")
@limiter.limit("30/minute")
async def health(request: Request) -> Dict[str, str]:
    return {"status": "ok"}


@router.post("/recommend", response_model=RecommendResponse)
@limiter.limit("10/minute")
async def recommend(request: Request, req: RecommendRequest) -> RecommendResponse:
    return await recommend_route(req)



