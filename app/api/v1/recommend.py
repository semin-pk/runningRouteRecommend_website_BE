from typing import Dict

from fastapi import APIRouter, Request, Depends
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

from app.api.deps import get_db
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
async def recommend(
    request: Request, 
    req: RecommendRequest,
    db: Session = Depends(get_db),
) -> RecommendResponse:
    """빠른 검색: 경유지 테마만 선택해서 한 가지 경로 추천 (리뷰 요약 포함)"""
    return await recommend_route(req, db=db)



