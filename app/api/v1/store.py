from typing import List, Dict
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
    RouteConfirmRequest,
    RouteCandidate,
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
    from uuid import uuid4
    from app.services.recommend_service import haversine_km, build_kakao_walk_url, calculate_destination_point
    import random
    
    route_candidates = []
    start_lat = search_req.start_lat if search_req.start_lat else search_req.latitude
    start_lng = search_req.start_lng if search_req.start_lng else search_req.longitude
    total_distance_km = search_req.total_distance_km if search_req.total_distance_km else 7.0  # 기본값 7km
    
    # 각 테마별로 가게 검색 (거리 기반)
    # 빠른 검색과 동일한 로직: 각 구간의 목표 거리를 계산하여 검색
    theme_stores: Dict[str, List[StoreInfo]] = {}
    
    # 각 구간의 목표 거리 계산 (빠른 검색과 동일한 방식)
    waypoint_count = len(themes)
    segment_distances = []
    if waypoint_count > 0:
        if search_req.is_round_trip:
            segment_count = waypoint_count + 1
            avg_distance = total_distance_km / segment_count
            segment_distances = [avg_distance] * waypoint_count
        else:
            avg_distance = total_distance_km / waypoint_count
            segment_distances = [avg_distance] * waypoint_count
    
    accumulated_distance = 0.0
    current_lat, current_lng = start_lat, start_lng
    
    for i, theme in enumerate(themes):
        segment_distance = segment_distances[i] if i < len(segment_distances) else segment_distances[-1] if segment_distances else total_distance_km / len(themes)
        
        # 목표 지점 계산 (현재 위치로부터 segment_distance만큼)
        random_bearing = random.uniform(0, 360)
        target_lat, target_lng = calculate_destination_point(
            current_lat, current_lng, segment_distance, random_bearing
        )
        
        # 출발점으로부터의 목표 거리 계산
        target_distance_from_start = accumulated_distance + segment_distance
        
        # 목표 거리 ± 1km 오차 범위 내에서 검색
        # 경로 다양성을 위해 각 테마당 5개씩 검색
        stores = await search_stores_by_theme(
            db=db,
            theme=theme,
            latitude=target_lat,
            longitude=target_lng,
            radius_m=1000,  # 오차 범위 1km
            limit=5,  # 다양성을 위해 5개로 증가
            target_distance_km=target_distance_from_start,
            start_lat=start_lat,
            start_lng=start_lng,
        )
        theme_stores[theme] = stores
        
        # 다음 경유지를 위한 현재 위치 업데이트 (평균 거리로 추정)
        # 실제로는 검색된 가게 중 하나의 위치를 사용하지만, 
        # 여기서는 목표 지점을 사용하여 다음 검색 위치 계산
        current_lat, current_lng = target_lat, target_lng
        accumulated_distance += segment_distance
    
    # 경로 생성: 빠른 검색과 동일한 방식으로 각 테마당 하나씩 랜덤 선택하여 경로 구성
    # (목표 거리 필터링은 search_stores_by_theme에서 이미 수행됨)
    from itertools import product
    
    # 각 테마별 가게 리스트
    theme_store_lists = [theme_stores.get(theme, []) for theme in themes]
    
    # 빠른 검색과 동일: 검색된 가게가 없는 테마가 있으면 빈 경로 반환
    if any(len(stores) == 0 for stores in theme_store_lists):
        print(f"[상세 검색] 경로 생성 실패: 일부 테마에서 가게를 찾을 수 없습니다.")
        return StoreSearchResponse(search_type="route", stores=None, routes=[])
    
    max_routes = 3
    route_count = 0
    used_store_combinations = set()
    
    print(f"[상세 검색] 경로 생성 시작: 테마 {len(themes)}개, 각 테마당 가게 {[len(stores) for stores in theme_store_lists]}")
    
    # 모든 조합 생성 후 거리 오차순 정렬
    all_combinations = []
    for store_combination in product(*theme_store_lists):
        if len(store_combination) != len(themes):
            continue
        
        # 경로 내 중복 가게 체크
        store_ids_in_route = [store.store_id for store in store_combination]
        if len(store_ids_in_route) != len(set(store_ids_in_route)):
            continue
        
        # 총 거리 계산
        total_distance = 0.0
        prev_lat = start_lat
        prev_lng = start_lng
        for store in store_combination:
            distance = haversine_km(prev_lat, prev_lng, float(store.latitude), float(store.longitude))
            total_distance += distance
            prev_lat = float(store.latitude)
            prev_lng = float(store.longitude)
        
        if search_req.is_round_trip:
            last_store = store_combination[-1]
            return_distance = haversine_km(float(last_store.latitude), float(last_store.longitude), start_lat, start_lng)
            total_distance += return_distance
        
        all_combinations.append((store_combination, total_distance))
    
    # 목표 거리 오차가 작은 순으로 정렬 (빠른 검색과 동일하게 거리 필터링 없이 오차순 정렬)
    all_combinations.sort(key=lambda x: abs(x[1] - total_distance_km))
    
    print(f"[상세 검색] 총 조합: {len(all_combinations)}개 (목표 거리: {total_distance_km}km)")
    
    # 상위 max_routes개 경로 생성 (다양성 확보)
    for store_combination, route_total_distance in all_combinations:
        if route_count >= max_routes:
            break
        
        store_ids_in_route = [store.store_id for store in store_combination]
        route_store_ids_set = frozenset(store_ids_in_route)
        
        if route_store_ids_set in used_store_combinations:
            continue
        
        # 경로 간 다양성 체크
        if len(used_store_combinations) > 0:
            has_diversity = False
            for used_combination in used_store_combinations:
                diff = route_store_ids_set.symmetric_difference(used_combination)
                if len(diff) >= 2:
                    has_diversity = True
                    break
            if not has_diversity:
                continue
        
        used_store_combinations.add(route_store_ids_set)
        
        route_id = str(uuid4())
        route_stores = []
        prev_lat = start_lat
        prev_lng = start_lng
        
        for i, store in enumerate(store_combination):
            has_review = check_review_summary_exists(db, store.store_id)
            review_summary = None
            if has_review:
                review = db.query(StoreReviewSummary).filter(
                    StoreReviewSummary.store_id == store.store_id
                ).first()
                if review:
                    review_summary = StoreReviewSummaryResponse.model_validate(review)
            
            distance = haversine_km(
                prev_lat, prev_lng,
                float(store.latitude), float(store.longitude)
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
            
            prev_lat = float(store.latitude)
            prev_lng = float(store.longitude)
        
        # 경로 URL 생성
        route_points = [{"name": "Start", "lat": f"{start_lat}", "lng": f"{start_lng}"}]
        for store in route_stores:
            route_points.append({
                "name": store.name,
                "lat": f"{store.latitude}",
                "lng": f"{store.longitude}",
            })
        if search_req.is_round_trip:
            route_points.append({"name": "Start", "lat": f"{start_lat}", "lng": f"{start_lng}"})
        
        route_url = build_kakao_walk_url(route_points) if route_points else None
        
        route_candidate = RouteCandidate(
            route_id=route_id,
            stores=route_stores,
            total_distance_km=round(route_total_distance, 2),
            route_url=route_url,
        )
        route_candidates.append(route_candidate)
        route_count += 1
    
    # 경로가 없으면 빈 배열 반환 (리뷰 크롤링 없이)
    if not route_candidates:
        print(f"[상세 검색] 경로 생성 실패: 유효한 경로가 없습니다. (검색된 가게 조합: {len(all_combinations)}개)")
        return StoreSearchResponse(search_type="route", stores=None, routes=[])
    
    # 경로 생성 성공 시에만 해당 경로의 가게들에 대해 리뷰 크롤링 실행
    processed_store_ids = set()
    for route in route_candidates:
        for store in route.stores:
            if store.store_id not in processed_store_ids:
                processed_store_ids.add(store.store_id)
                if not check_review_summary_exists(db, store.store_id):
                    # 해당 가게의 테마 찾기
                    store_theme = None
                    for theme, stores in theme_stores.items():
                        for s in stores:
                            if s.store_id == store.store_id:
                                store_theme = theme
                                break
                        if store_theme:
                            break
                    
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

