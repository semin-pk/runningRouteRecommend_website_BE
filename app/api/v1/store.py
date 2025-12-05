from typing import List
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.store import StoreInfo, StoreReviewSummary
from app.schemas.store import (
    StoreInfoCreate,
    StoreInfoResponse,
    StoreInfoUpdate,
    StoreReviewSummaryCreate,
    StoreReviewSummaryResponse,
    StoreReviewSummaryUpdate,
    StoreWithReviewResponse,
    StoreSearchRequest,
    StoreSearchResponse,
    StoreCandidateResponse,
    StoreConfirmRequest,
)
from app.services.store_service import (
    search_stores_by_theme,
    check_review_summary_exists,
    process_review_summary_background,
)
from app.services.recommend_service import recommend_route
from app.schemas.recommend import RecommendRequest, RecommendResponse

router = APIRouter()
limiter: Limiter = Limiter(key_func=get_remote_address)


@router.post("/stores", response_model=StoreInfoResponse, status_code=201)
@limiter.limit("30/minute")
async def create_store(
    request: Request,
    store_data: StoreInfoCreate,
    db: Session = Depends(get_db),
) -> StoreInfoResponse:
    """가게 정보 생성"""
    store_id = str(uuid4())
    db_store = StoreInfo(store_id=store_id, **store_data.model_dump())
    db.add(db_store)
    db.commit()
    db.refresh(db_store)
    return StoreInfoResponse.model_validate(db_store)


@router.get("/stores/{store_id}", response_model=StoreWithReviewResponse)
@limiter.limit("60/minute")
async def get_store(
    request: Request,
    store_id: str,
    db: Session = Depends(get_db),
) -> StoreWithReviewResponse:
    """가게 정보 조회 (리뷰 요약 포함)"""
    store = db.query(StoreInfo).filter(StoreInfo.store_id == store_id).first()
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")
    return StoreWithReviewResponse.model_validate(store)


@router.get("/stores", response_model=List[StoreInfoResponse])
@limiter.limit("60/minute")
async def list_stores(
    request: Request,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
) -> List[StoreInfoResponse]:
    """가게 목록 조회"""
    stores = db.query(StoreInfo).offset(skip).limit(limit).all()
    return [StoreInfoResponse.model_validate(store) for store in stores]


@router.put("/stores/{store_id}", response_model=StoreInfoResponse)
@limiter.limit("30/minute")
async def update_store(
    request: Request,
    store_id: str,
    store_data: StoreInfoUpdate,
    db: Session = Depends(get_db),
) -> StoreInfoResponse:
    """가게 정보 업데이트"""
    store = db.query(StoreInfo).filter(StoreInfo.store_id == store_id).first()
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")
    
    update_data = store_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(store, field, value)
    
    db.commit()
    db.refresh(store)
    return StoreInfoResponse.model_validate(store)


@router.delete("/stores/{store_id}", status_code=204)
@limiter.limit("30/minute")
async def delete_store(
    request: Request,
    store_id: str,
    db: Session = Depends(get_db),
):
    """가게 정보 삭제"""
    store = db.query(StoreInfo).filter(StoreInfo.store_id == store_id).first()
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")
    db.delete(store)
    db.commit()


@router.post("/stores/{store_id}/review", response_model=StoreReviewSummaryResponse, status_code=201)
@limiter.limit("30/minute")
async def create_store_review(
    request: Request,
    store_id: str,
    review_data: StoreReviewSummaryCreate,
    db: Session = Depends(get_db),
) -> StoreReviewSummaryResponse:
    """가게 리뷰 요약 생성"""
    # 가게 존재 확인
    store = db.query(StoreInfo).filter(StoreInfo.store_id == store_id).first()
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")
    
    # 이미 리뷰가 있는지 확인
    existing_review = db.query(StoreReviewSummary).filter(
        StoreReviewSummary.store_id == store_id
    ).first()
    if existing_review:
        raise HTTPException(status_code=400, detail="Review summary already exists for this store")
    
    db_review = StoreReviewSummary(
        store_id=store_id,
        **review_data.model_dump(exclude={"store_id"})
    )
    db.add(db_review)
    db.commit()
    db.refresh(db_review)
    return StoreReviewSummaryResponse.model_validate(db_review)


@router.get("/stores/{store_id}/review", response_model=StoreReviewSummaryResponse)
@limiter.limit("60/minute")
async def get_store_review(
    request: Request,
    store_id: str,
    db: Session = Depends(get_db),
) -> StoreReviewSummaryResponse:
    """가게 리뷰 요약 조회"""
    review = db.query(StoreReviewSummary).filter(
        StoreReviewSummary.store_id == store_id
    ).first()
    if not review:
        raise HTTPException(status_code=404, detail="Review summary not found")
    return StoreReviewSummaryResponse.model_validate(review)


@router.put("/stores/{store_id}/review", response_model=StoreReviewSummaryResponse)
@limiter.limit("30/minute")
async def update_store_review(
    request: Request,
    store_id: str,
    review_data: StoreReviewSummaryUpdate,
    db: Session = Depends(get_db),
) -> StoreReviewSummaryResponse:
    """가게 리뷰 요약 업데이트"""
    review = db.query(StoreReviewSummary).filter(
        StoreReviewSummary.store_id == store_id
    ).first()
    if not review:
        raise HTTPException(status_code=404, detail="Review summary not found")
    
    update_data = review_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(review, field, value)
    
    db.commit()
    db.refresh(review)
    return StoreReviewSummaryResponse.model_validate(review)


@router.post("/stores/search", response_model=StoreSearchResponse)
@limiter.limit("30/minute")
async def search_stores(
    request: Request,
    search_req: StoreSearchRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> StoreSearchResponse:
    """
    테마로 가게를 검색하고, 상위 3개를 DB에 저장/업데이트.
    각 가게의 리뷰 요약 상태를 확인하고, 없으면 비동기로 크롤링/요약 처리.
    """
    # Kakao API로 검색하고 DB에 저장
    stores = await search_stores_by_theme(
        db=db,
        theme=search_req.theme,
        latitude=search_req.latitude,
        longitude=search_req.longitude,
        radius_m=search_req.radius_m,
        limit=3,
    )
    
    if not stores:
        return StoreSearchResponse(stores=[])
    
    # 각 가게의 리뷰 요약 상태 확인 및 비동기 처리
    store_candidates = []
    for store in stores:
        # 리뷰 요약 존재 여부 확인
        has_review = check_review_summary_exists(db, store.store_id)
        
        review_summary = None
        if has_review:
            review = db.query(StoreReviewSummary).filter(
                StoreReviewSummary.store_id == store.store_id
            ).first()
            if review:
                review_summary = StoreReviewSummaryResponse.model_validate(review)
        
        # 리뷰 요약이 없으면 비동기로 처리 시작
        summary_status = "ready" if has_review else "processing"
        
        if not has_review:
            # BackgroundTasks로 크롤링 및 요약 처리
            background_tasks.add_task(
                process_review_summary_background,
                store_id=store.store_id,
                store_name=store.name,
            )
        
        store_candidate = StoreCandidateResponse(
            store_id=store.store_id,
            name=store.name,
            address=store.address,
            longitude=store.longitude,
            latitude=store.latitude,
            phone=store.phone,
            open_time=store.open_time,
            close_time=store.close_time,
            summary_status=summary_status,
            review_summary=review_summary,
        )
        store_candidates.append(store_candidate)
    
    return StoreSearchResponse(stores=store_candidates)


@router.post("/stores/confirm", response_model=RecommendResponse)
@limiter.limit("30/minute")
async def confirm_store(
    request: Request,
    confirm_req: StoreConfirmRequest,
    db: Session = Depends(get_db),
) -> RecommendResponse:
    """
    선택된 가게로 경로를 확정하고 추천 결과를 반환.
    """
    # 가게 존재 확인
    store = db.query(StoreInfo).filter(StoreInfo.store_id == confirm_req.store_id).first()
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")
    
    # 기존 recommend_route 함수를 사용하여 경로 생성
    # 가게 위치를 목적지로 설정
    recommend_req = RecommendRequest(
        start_lat=confirm_req.start_lat,
        start_lng=confirm_req.start_lng,
        total_distance_km=confirm_req.total_distance_km,
        waypoints=[],  # 단일 목적지이므로 waypoint 없음
        is_round_trip=confirm_req.is_round_trip,
    )
    
    # 경로 추천 수행 (실제로는 선택된 가게로 직접 가는 경로를 만들어야 함)
    # 여기서는 간단히 선택된 가게를 목적지로 하는 경로를 생성
    from app.services.recommend_service import build_kakao_walk_url, haversine_km
    
    store_lat = float(store.latitude)
    store_lng = float(store.longitude)
    distance_km = haversine_km(
        confirm_req.start_lat,
        confirm_req.start_lng,
        store_lat,
        store_lng,
    )
    
    route_points = [
        {"name": "Start", "lat": f"{confirm_req.start_lat}", "lng": f"{confirm_req.start_lng}"},
        {"name": store.name, "lat": f"{store_lat}", "lng": f"{store_lng}"},
    ]
    
    if confirm_req.is_round_trip:
        route_points.append(
            {"name": "Start", "lat": f"{confirm_req.start_lat}", "lng": f"{confirm_req.start_lng}"}
        )
        distance_km *= 2
    
    route_url = build_kakao_walk_url(route_points)
    
    from app.schemas.recommend import WaypointResult
    
    waypoint_result = WaypointResult(
        place_name=store.name,
        address_name=store.address,
        road_address_name=None,
        phone=store.phone,
        place_url=None,
        category_name=None,
        x=str(store_lng),
        y=str(store_lat),
        distance_km=distance_km,
        theme_keyword="",
        order=1,
    )
    
    return RecommendResponse(
        waypoints=[waypoint_result],
        route_url=route_url,
        total_distance_km=confirm_req.total_distance_km,
        actual_total_distance_km=round(distance_km, 2),
        is_round_trip=confirm_req.is_round_trip,
        candidates_considered=1,
    )

