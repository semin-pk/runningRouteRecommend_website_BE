from typing import List, Dict
from uuid import uuid4
from decimal import Decimal, ROUND_HALF_UP

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
    RouteConfirmRequest,
    RouteCandidate,
)
from app.services.store_service import (
    search_stores_by_theme,
    check_review_summary_exists,
    process_review_summary_background,
    find_or_create_store,
)
from app.services.recommend_service import recommend_route
from app.schemas.recommend import RecommendRequest, RecommendResponse, Waypoint

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
    상세 검색: 여러 테마(경유지)로 가게를 검색.
    - 경유지 1개: 가게 선택 가능
    - 경유지 2개 이상: 경로 선택 가능
    """
    themes = search_req.themes
    
    # 경유지가 1개일 때: 거리 기반 검색 (가게 선택)
    if len(themes) == 1:
        theme = themes[0]
        start_lat = search_req.start_lat if search_req.start_lat else search_req.latitude
        start_lng = search_req.start_lng if search_req.start_lng else search_req.longitude
        total_distance_km = search_req.total_distance_km if search_req.total_distance_km else 7.0
        
        # 왕복인 경우 편도 거리 계산
        target_distance = total_distance_km / 2 if search_req.is_round_trip else total_distance_km
        
        # 목표 거리만큼 떨어진 지점 계산 (랜덤 방향)
        from app.services.recommend_service import calculate_destination_point
        import random
        random_bearing = random.uniform(0, 360)
        target_lat, target_lng = calculate_destination_point(
            start_lat, start_lng, target_distance, random_bearing
        )
        
        stores = await search_stores_by_theme(
            db=db,
            theme=theme,
            latitude=target_lat,
            longitude=target_lng,
            radius_m=search_req.radius_m,
            limit=3,
            target_distance_km=target_distance,
            start_lat=start_lat,
            start_lng=start_lng,
        )
        
        if not stores:
            return StoreSearchResponse(search_type="single", stores=[], routes=None)
        
        # 각 가게의 리뷰 요약 상태 확인 및 비동기 처리
        store_candidates = []
        for store in stores:
            has_review = check_review_summary_exists(db, store.store_id)
            
            review_summary = None
            if has_review:
                review = db.query(StoreReviewSummary).filter(
                    StoreReviewSummary.store_id == store.store_id
                ).first()
                if review:
                    review_summary = StoreReviewSummaryResponse.model_validate(review)
            
            summary_status = "ready" if has_review else "processing"
            
            if not has_review:
                background_tasks.add_task(
                    process_review_summary_background,
                    store_id=store.store_id,
                    store_name=store.name,
                    store_address=store.address,
                    theme=theme,
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
        
        print(f"[상세 검색] 단일 테마 검색 완료: {len(store_candidates)}개 가게")
        return StoreSearchResponse(search_type="single", stores=store_candidates, routes=None)
    
    # 경유지가 2개 이상일 때: 경로 생성
    # 빠른 검색의 recommend_route 함수를 3번 호출하여 3개의 경로 생성
    from app.services.recommend_service import build_kakao_walk_url, haversine_km
    
    start_lat = search_req.start_lat if search_req.start_lat else search_req.latitude
    start_lng = search_req.start_lng if search_req.start_lng else search_req.longitude
    total_distance_km = search_req.total_distance_km if search_req.total_distance_km else 7.0  # 기본값 7km
    
    # 테마를 waypoint 형식으로 변환
    waypoints = [Waypoint(theme_keyword=theme, order=i+1) for i, theme in enumerate(themes)]
    
    # 빠른 검색의 recommend_route를 3번 호출하여 다양한 경로 생성
    route_candidates = []
    used_place_names = set()  # 이미 사용된 장소 이름 추적 (다양성 확보)
    max_routes = 3
    
    print(f"[상세 검색] 경로 생성 시작: 테마 {len(themes)}개, 빠른 검색 방식으로 {max_routes}개 경로 생성")
    
    for route_idx in range(max_routes):
        try:
            # RecommendRequest 생성
            recommend_req = RecommendRequest(
                start_lat=start_lat,
                start_lng=start_lng,
                total_distance_km=total_distance_km,
                waypoints=waypoints,
                is_round_trip=search_req.is_round_trip,
            )
            
            # 빠른 검색의 recommend_route 호출
            recommend_response = await recommend_route(recommend_req, db=db)
            
            # waypoint가 없으면 실패
            if not recommend_response.waypoints:
                print(f"[상세 검색] 경로 {route_idx + 1} 생성 실패: waypoint 없음")
                continue
            
            # 이미 사용된 장소가 모두 포함되어 있는지 확인 (다양성 체크)
            route_place_names = {w.place_name for w in recommend_response.waypoints}
            if route_place_names.issubset(used_place_names) and len(used_place_names) > 0:
                # 모든 장소가 이미 사용되었으면 스킵
                print(f"[상세 검색] 경로 {route_idx + 1} 스킵: 다양성 부족")
                continue
            
            # 사용된 장소 목록에 추가
            used_place_names.update(route_place_names)
            
            # RecommendResponse를 RouteCandidate로 변환
            route_stores = []
            for waypoint in recommend_response.waypoints:
                # waypoint 정보를 dict 형식으로 변환하여 find_or_create_store 사용
                kakao_result_dict = {
                    "place_name": waypoint.place_name,
                    "address_name": waypoint.address_name or "",
                    "road_address_name": waypoint.road_address_name or "",
                    "x": waypoint.x,
                    "y": waypoint.y,
                    "phone": waypoint.phone or "",
                }
                
                # DB에 가게 저장 또는 조회
                store = find_or_create_store(db, kakao_result_dict)
                
                has_review = False
                review_summary = None
                
                if check_review_summary_exists(db, store.store_id):
                    has_review = True
                    review = db.query(StoreReviewSummary).filter(
                        StoreReviewSummary.store_id == store.store_id
                    ).first()
                    if review:
                        review_summary = StoreReviewSummaryResponse.model_validate(review)
                elif waypoint.review_summary:
                    # waypoint에 리뷰 요약이 있으면 사용
                    has_review = True
                    review_summary = StoreReviewSummaryResponse(
                        main_menu=waypoint.review_summary.get("main_menu"),
                        atmosphere=waypoint.review_summary.get("atmosphere"),
                        recommended_for=waypoint.review_summary.get("recommended_for"),
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
                    summary_status="ready" if has_review else "processing",
                    review_summary=review_summary,
                )
                
                route_stores.append(store_candidate)
            
            route_candidate = RouteCandidate(
                route_id=str(uuid4()),
                stores=route_stores,
                total_distance_km=recommend_response.actual_total_distance_km or round(recommend_response.total_distance_km, 2),
                route_url=recommend_response.route_url,
            )
            route_candidates.append(route_candidate)
            print(f"[상세 검색] 경로 {route_idx + 1} 생성 완료: {len(route_stores)}개 가게, 총 거리 {route_candidate.total_distance_km}km")
            
        except Exception as e:
            print(f"[상세 검색] 경로 {route_idx + 1} 생성 중 오류: {e}")
            continue
    
    # 경로가 없으면 빈 배열 반환 (리뷰 크롤링 없이)
    if not route_candidates:
        print(f"[상세 검색] 경로 생성 실패: 유효한 경로가 없습니다.")
        return StoreSearchResponse(search_type="route", stores=None, routes=[])
    
    # 경로 생성 성공 시에만 해당 경로의 가게들에 대해 리뷰 크롤링 실행
    processed_store_ids = set()
    for route in route_candidates:
        for store in route.stores:
            if store.store_id not in processed_store_ids:
                processed_store_ids.add(store.store_id)
                if not check_review_summary_exists(db, store.store_id):
                    # 해당 가게의 테마 찾기 (waypoint에서)
                    store_theme = None
                    for waypoint in waypoints:
                        if waypoint.theme_keyword in store.name or any(waypoint.theme_keyword in theme for theme in themes):
                            store_theme = waypoint.theme_keyword
                            break
                    if not store_theme and themes:
                        store_theme = themes[0]  # 기본값
                    
                    background_tasks.add_task(
                        process_review_summary_background,
                        store_id=store.store_id,
                        store_name=store.name,
                        store_address=store.address,
                        theme=store_theme,
                    )
    
    print(f"[상세 검색] 경로 생성 완료: {len(route_candidates)}개 경로 생성, 리뷰 크롤링 대상: {len(processed_store_ids)}개 가게")
    return StoreSearchResponse(search_type="route", stores=None, routes=route_candidates)


@router.post("/stores/confirm", response_model=RecommendResponse)
@limiter.limit("30/minute")
async def confirm_store(
    request: Request,
    confirm_req: StoreConfirmRequest,
    db: Session = Depends(get_db),
) -> RecommendResponse:
    """
    단일 가게 선택 확정: 선택된 가게로 경로를 확정하고 추천 결과를 반환.
    """
    # 가게 존재 확인
    store = db.query(StoreInfo).filter(StoreInfo.store_id == confirm_req.store_id).first()
    if not store:
        raise HTTPException(status_code=404, detail="Store not found")
    
    from app.services.recommend_service import build_kakao_walk_url, haversine_km
    from app.schemas.recommend import WaypointResult
    
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
    
    # 리뷰 요약 조회
    review_summary = None
    review = db.query(StoreReviewSummary).filter(
        StoreReviewSummary.store_id == store.store_id
    ).first()
    if review:
        review_summary = {
            "main_menu": review.main_menu,
            "atmosphere": review.atmosphere,
            "recommended_for": review.recommended_for,
        }
    
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
        review_summary=review_summary,
    )
    
    return RecommendResponse(
        waypoints=[waypoint_result],
        route_url=route_url,
        total_distance_km=confirm_req.total_distance_km,
        actual_total_distance_km=round(distance_km, 2),
        is_round_trip=confirm_req.is_round_trip,
        candidates_considered=1,
    )


@router.post("/routes/confirm", response_model=RecommendResponse)
@limiter.limit("30/minute")
async def confirm_route(
    request: Request,
    confirm_req: RouteConfirmRequest,
    db: Session = Depends(get_db),
) -> RecommendResponse:
    """
    경로 선택 확정: 선택된 경로로 확정하고 추천 결과를 반환.
    store_ids 리스트로 경로의 가게들을 순서대로 전달받음.
    """
    from app.services.recommend_service import build_kakao_walk_url, haversine_km
    from app.schemas.recommend import WaypointResult
    
    waypoint_results = []
    route_points = [
        {"name": "Start", "lat": f"{confirm_req.start_lat}", "lng": f"{confirm_req.start_lng}"}
    ]
    
    prev_lat = confirm_req.start_lat
    prev_lng = confirm_req.start_lng
    total_distance = 0.0
    
    for order, store_id in enumerate(confirm_req.store_ids, start=1):
        store = db.query(StoreInfo).filter(StoreInfo.store_id == store_id).first()
        if not store:
            raise HTTPException(status_code=404, detail=f"Store {store_id} not found")
        
        store_lat = float(store.latitude)
        store_lng = float(store.longitude)
        distance_km = haversine_km(prev_lat, prev_lng, store_lat, store_lng)
        total_distance += distance_km
        
        # 리뷰 요약 조회
        review_summary = None
        review = db.query(StoreReviewSummary).filter(
            StoreReviewSummary.store_id == store.store_id
        ).first()
        if review:
            review_summary = {
                "main_menu": review.main_menu,
                "atmosphere": review.atmosphere,
                "recommended_for": review.recommended_for,
            }
        
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
            order=order,
            review_summary=review_summary,
        )
        waypoint_results.append(waypoint_result)
        
        route_points.append({
            "name": store.name,
            "lat": f"{store_lat}",
            "lng": f"{store_lng}",
        })
        
        prev_lat = store_lat
        prev_lng = store_lng
    
    # 왕복 거리 추가
    if confirm_req.is_round_trip and waypoint_results:
        return_distance = haversine_km(
            prev_lat, prev_lng,
            confirm_req.start_lat, confirm_req.start_lng,
        )
        total_distance += return_distance
        route_points.append({
            "name": "Start",
            "lat": f"{confirm_req.start_lat}",
            "lng": f"{confirm_req.start_lng}",
        })
    
    route_url = build_kakao_walk_url(route_points)
    
    return RecommendResponse(
        waypoints=waypoint_results,
        route_url=route_url,
        total_distance_km=confirm_req.total_distance_km,
        actual_total_distance_km=round(total_distance, 2),
        is_round_trip=confirm_req.is_round_trip,
        candidates_considered=len(confirm_req.store_ids),
    )

