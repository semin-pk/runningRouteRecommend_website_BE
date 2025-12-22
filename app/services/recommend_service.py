import math
import random
from typing import Any, Dict, List, Tuple, Optional

import httpx
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.core.config import KAKAO_REST_API_KEY
from app.schemas.recommend import (
    RecommendRequest,
    RecommendResponse,
    WaypointResult,
)
from app.models.store import StoreInfo, StoreReviewSummary


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points on Earth (km)."""
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def calculate_destination_point(
    start_lat: float, start_lng: float, distance_km: float, bearing_deg: float
) -> Tuple[float, float]:
    """Calculate destination coordinates from start point, distance and bearing."""
    r = 6371.0

    bearing_rad = math.radians(bearing_deg)
    lat1_rad = math.radians(start_lat)
    lng1_rad = math.radians(start_lng)

    lat2_rad = math.asin(
        math.sin(lat1_rad) * math.cos(distance_km / r)
        + math.cos(lat1_rad)
        * math.sin(distance_km / r)
        * math.cos(bearing_rad)
    )

    lng2_rad = lng1_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(distance_km / r) * math.cos(lat1_rad),
        math.cos(distance_km / r) - math.sin(lat1_rad) * math.sin(lat2_rad),
    )

    dest_lat = math.degrees(lat2_rad)
    dest_lng = math.degrees(lng2_rad)
    return dest_lat, dest_lng


def build_kakao_walk_url(points: List[Dict[str, str]]) -> str:
    """Build Kakao map walk route URL using /link/by/walk pattern."""
    parts = []
    for p in points:
        name = p.get("name", "Point")
        lat = p["lat"]
        lng = p["lng"]
        parts.append(f"{name},{lat},{lng}")
    return "https://map.kakao.com/link/by/walk/" + "/".join(parts)


async def kakao_keyword_search(
    query: str,
    x: float,
    y: float,
    radius_m: int,
    page_limit: int = 30,
) -> List[Dict[str, Any]]:
    """Query Kakao Local Keyword Search API around a point within radius."""
    if not KAKAO_REST_API_KEY:
        raise HTTPException(
            status_code=500, detail="KAKAO_REST_API_KEY is not configured"
        )

    headers = {"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"}
    url = "https://dapi.kakao.com/v2/local/search/keyword"
    
    # 키워드에 따라 category_group_code 설정
    category_group_code = None
    if "카페" in query:
        category_group_code = "CE7"
    elif "맛집" in query or "음식" in query or "식당" in query:
        category_group_code = "FD6"

    async with httpx.AsyncClient(timeout=10.0) as client:
        params = {
            "query": query,
            "x": x,
            "y": y,
            "radius": radius_m,
            "sort": "distance",
            "page": page_limit,
            "size": 15,
        }
        if category_group_code:
            params["category_group_code"] = category_group_code
        r = await client.get(url, headers=headers, params=params)
        if r.status_code != 200:
            error_text = r.text
            if "NotAuthorizedError" in error_text and "OPEN_MAP_AND_LOCAL" in error_text:
                raise HTTPException(
                    status_code=403,
                    detail=(
                        "Kakao Local API service is not enabled. "
                        "Please enable 'OPEN_MAP_AND_LOCAL' service in your Kakao Developers console."
                    ),
                )
            raise HTTPException(status_code=502, detail=f"Kakao API error: {error_text}")
        data = r.json()
        return data.get("documents", [])


async def find_waypoint_places(
    keyword: str,
    center_lat: float,
    center_lng: float,
    search_radius_m: int = 2000,
    target_distance_km: Optional[float] = None,
    start_lat: Optional[float] = None,
    start_lng: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Search places around a waypoint using Kakao API.
    If target_distance_km and start coordinates are provided, filters places
    that are approximately target_distance_km away from start (within 1km error).
    """
    places = await kakao_keyword_search(
        keyword,
        x=center_lng,
        y=center_lat,
        radius_m=search_radius_m,
        page_limit=30,
    )

    # 허용된 카테고리 (프랜차이즈, 스터디카페, 무인카페 등 제외)
    allowed_categories = [
        "음식점 > 카페",
        "음식점 > 카페 > 커피전문점",
        "음식점 > 카페 > 테마카페 > 디저트카페",
    ]

    scored: List[Dict[str, Any]] = []
    for p in places:
        try:
            lat = float(p["y"])
            lng = float(p["x"])
        except Exception:
            continue
        
        # 카테고리 필터링: 허용된 카테고리만 포함
        category_name = p.get("category_name", "")
        if category_name and not any(category_name == allowed for allowed in allowed_categories):
            continue

        # 중심점으로부터의 거리
        d_km = haversine_km(center_lat, center_lng, lat, lng)
        p_copy = dict(p)
        p_copy["distance_km"] = d_km
        
        # 출발점으로부터의 거리 계산 (목표 거리 필터링용)
        if target_distance_km is not None and start_lat is not None and start_lng is not None:
            distance_from_start = haversine_km(start_lat, start_lng, lat, lng)
            p_copy["distance_from_start_km"] = distance_from_start
            
            # 목표 거리 ± 1km 오차 범위 내에 있는지 확인
            error_margin_km = 1.0
            if abs(distance_from_start - target_distance_km) <= error_margin_km:
                p_copy["distance_error"] = abs(distance_from_start - target_distance_km)
                scored.append(p_copy)
        else:
            scored.append(p_copy)

    # 목표 거리가 있으면 오차가 작은 순으로, 없으면 거리 순으로 정렬
    if target_distance_km is not None and start_lat is not None and start_lng is not None:
        scored.sort(key=lambda x: x.get("distance_error", float('inf')))
    else:
        scored.sort(key=lambda x: x["distance_km"])
    
    return scored


def distribute_distance_for_waypoints(
    total_distance_km: float,
    waypoint_count: int,
    is_round_trip: bool,
) -> List[float]:
    """Distribute total distance across waypoints."""
    if waypoint_count == 0:
        return []

    if is_round_trip:
        segment_count = waypoint_count + 1
        avg_distance = total_distance_km / segment_count
        distances = [avg_distance] * waypoint_count
    else:
        avg_distance = total_distance_km / waypoint_count
        distances = [avg_distance] * waypoint_count

    return distances


def get_review_summary_for_place(
    db: Session,
    place_name: str,
    place_address: Optional[str],
    place_lat: float,
    place_lng: float,
) -> Optional[Dict[str, Any]]:
    """가게 정보로 리뷰 요약을 찾아서 반환"""
    try:
        # 이름과 주소로 가게 찾기
        query = db.query(StoreInfo).filter(StoreInfo.name == place_name)
        if place_address:
            query = query.filter(StoreInfo.address == place_address)
        
        store = query.first()
        if not store:
            return None
        
        # 리뷰 요약 찾기
        review = db.query(StoreReviewSummary).filter(
            StoreReviewSummary.store_id == store.store_id
        ).first()
        
        if review:
            return {
                "main_menu": review.main_menu,
                "atmosphere": review.atmosphere,
                "recommended_for": review.recommended_for,
            }
        return None
    except Exception as e:
        print(f"[ERROR] 리뷰 요약 조회 실패: {e}")
        return None


async def recommend_route(req: RecommendRequest, db: Optional[Session] = None) -> RecommendResponse:
    """Main business logic for route recommendation."""
    start_lat = req.start_lat
    start_lng = req.start_lng
    total_distance_km = req.total_distance_km
    waypoints = req.waypoints
    is_round_trip = req.is_round_trip

    # case: no waypoints
    if not waypoints:
        # 왕복인 경우 편도 거리 계산
        target_distance = total_distance_km / 2 if is_round_trip else total_distance_km
        
        # 목표 거리만큼 떨어진 지점 계산 (랜덤 방향)
        random_bearing = random.uniform(0, 360)
        target_lat, target_lng = calculate_destination_point(
            start_lat, start_lng, target_distance, random_bearing
        )

        # 목표 거리 ± 1km 오차 범위 내에서 카페 검색
        places = await find_waypoint_places(
            keyword="카페",
            center_lat=target_lat,
            center_lng=target_lng,
            search_radius_m=1000,  # 오차 범위 1km = 1000m
            target_distance_km=target_distance,
            start_lat=start_lat,
            start_lng=start_lng,
        )

        if not places:
            # 필터링된 결과가 없으면 오차 범위를 넓혀서 재시도
            places = await kakao_keyword_search(
                "카페",
                x=target_lng,
                y=target_lat,
                radius_m=2000,  # 오차 범위를 2km로 확대
                page_limit=30,
            )
            
            if not places:
                route_url = build_kakao_walk_url(
                    [
                        {
                            "name": "Start",
                            "lat": f"{start_lat}",
                            "lng": f"{start_lng}",
                        }
                    ]
                )
                return RecommendResponse(
                    waypoints=[],
                    route_url=route_url,
                    total_distance_km=total_distance_km,
                    actual_total_distance_km=0,
                    is_round_trip=is_round_trip,
                    candidates_considered=0,
                )
            
            # 카테고리 필터링 (프랜차이즈, 스터디카페, 무인카페 등 제외)
            allowed_categories = [
                "음식점 > 카페",
                "음식점 > 카페 > 커피전문점",
                "음식점 > 카페 > 테마카페 > 디저트카페",
            ]
            places = [p for p in places if p.get("category_name", "") in allowed_categories]
            
            if not places:
                route_url = build_kakao_walk_url(
                    [
                        {
                            "name": "Start",
                            "lat": f"{start_lat}",
                            "lng": f"{start_lng}",
                        }
                    ]
                )
                return RecommendResponse(
                    waypoints=[],
                    route_url=route_url,
                    total_distance_km=total_distance_km,
                    actual_total_distance_km=0,
                    is_round_trip=is_round_trip,
                    candidates_considered=0,
                )
            
            # 거리 계산 및 필터링
            filtered_places = []
            for p in places:
                try:
                    lat = float(p["y"])
                    lng = float(p["x"])
                    distance_from_start = haversine_km(start_lat, start_lng, lat, lng)
                    error_margin_km = 1.5  # 확대된 오차 범위
                    if abs(distance_from_start - target_distance) <= error_margin_km:
                        p_copy = dict(p)
                        p_copy["distance_from_start_km"] = distance_from_start
                        p_copy["distance_error"] = abs(distance_from_start - target_distance)
                        filtered_places.append(p_copy)
                except Exception:
                    continue
            
            if filtered_places:
                filtered_places.sort(key=lambda x: x.get("distance_error", float('inf')))
                places = filtered_places
            else:
                # 여전히 없으면 가장 가까운 것 선택
                places = [dict(p, **{"distance_from_start_km": haversine_km(start_lat, start_lng, float(p["y"]), float(p["x"]))}) for p in places]
                places.sort(key=lambda x: abs(x.get("distance_from_start_km", float('inf')) - target_distance))
        
        # 랜덤으로 선택 (오차가 작은 상위 5개 중에서)
        top_candidates = places[:min(5, len(places))]
        selected = random.choice(top_candidates) if top_candidates else places[0]
        dest_name = selected.get("place_name") or "Destination"
        dest_lat = f"{selected['y']}"
        dest_lng = f"{selected['x']}"

        route_points = [
            {"name": "Start", "lat": f"{start_lat}", "lng": f"{start_lng}"},
            {"name": dest_name, "lat": dest_lat, "lng": dest_lng},
        ]

        if is_round_trip:
            route_points.append(
                {"name": "Start", "lat": f"{start_lat}", "lng": f"{start_lng}"}
            )

        route_url = build_kakao_walk_url(route_points)

        # 리뷰 요약 조회
        review_summary = None
        if db:
            review_summary = get_review_summary_for_place(
                db=db,
                place_name=dest_name,
                place_address=selected.get("address_name") or selected.get("road_address_name"),
                place_lat=float(selected["y"]),
                place_lng=float(selected["x"]),
            )
        
        actual_distance = haversine_km(
            start_lat, start_lng, float(selected["y"]), float(selected["x"])
        )
        
        waypoint_result = WaypointResult(
            place_name=dest_name,
            address_name=selected.get("address_name"),
            road_address_name=selected.get("road_address_name"),
            phone=selected.get("phone"),
            place_url=selected.get("place_url"),
            category_name=selected.get("category_name"),
            x=selected.get("x"),
            y=selected.get("y"),
            distance_km=actual_distance,
            theme_keyword="카페",
            order=1,
            review_summary=review_summary,
        )

        actual_total_distance = actual_distance
        if is_round_trip:
            actual_total_distance += actual_distance

        return RecommendResponse(
            waypoints=[waypoint_result],
            route_url=route_url,
            total_distance_km=total_distance_km,
            actual_total_distance_km=round(actual_total_distance, 2),
            is_round_trip=is_round_trip,
            candidates_considered=len(places),
        )

    # case: with waypoints
    waypoint_count = len(waypoints)
    segment_distances = distribute_distance_for_waypoints(
        total_distance_km, waypoint_count, is_round_trip
    )

    waypoint_results: List[WaypointResult] = []
    route_points: List[Dict[str, str]] = [
        {"name": "Start", "lat": f"{start_lat}", "lng": f"{start_lng}"}
    ]
    current_lat, current_lng = start_lat, start_lng
    total_candidates = 0
    accumulated_distance = 0.0  # 누적 거리 추적
    used_place_names = set()  # 이미 사용된 장소 이름 추적 (중복 방지)

    for i, waypoint in enumerate(sorted(waypoints, key=lambda w: w.order)):
        # 각 구간의 목표 거리 계산
        # 남은 거리를 남은 구간 수로 나눔 (더 정확한 분배)
        remaining_segments = waypoint_count - i
        if is_round_trip:
            remaining_segments += 1  # 돌아오는 구간 포함
        
        remaining_distance = total_distance_km - accumulated_distance
        if is_round_trip and i == waypoint_count - 1:
            # 마지막 waypoint는 돌아오는 거리도 고려
            segment_distance = remaining_distance / 2
        else:
            segment_distance = remaining_distance / remaining_segments if remaining_segments > 0 else remaining_distance
        
        # 목표 지점 계산 (현재 위치로부터 segment_distance만큼)
        random_bearing = random.uniform(0, 360)
        target_lat, target_lng = calculate_destination_point(
            current_lat, current_lng, segment_distance, random_bearing
        )

        # 출발점으로부터의 목표 거리 계산
        target_distance_from_start = accumulated_distance + segment_distance
        
        # 목표 거리 ± 1km 오차 범위 내에서 검색
        places = await find_waypoint_places(
            keyword=waypoint.theme_keyword,
            center_lat=target_lat,
            center_lng=target_lng,
            search_radius_m=1000,  # 오차 범위 1km
            target_distance_km=target_distance_from_start,
            start_lat=start_lat,
            start_lng=start_lng,
        )

        if not places:
            # 필터링된 결과가 없으면 오차 범위를 넓혀서 재시도
            places_raw = await kakao_keyword_search(
                waypoint.theme_keyword,
                x=target_lng,
                y=target_lat,
                radius_m=2000,
                page_limit=30,
            )
            
            # 카테고리 필터링 (프랜차이즈, 스터디카페, 무인카페 등 제외) - 카페인 경우에만
            if "카페" in waypoint.theme_keyword:
                allowed_categories = [
                    "음식점 > 카페",
                    "음식점 > 카페 > 커피전문점",
                    "음식점 > 카페 > 테마카페 > 디저트카페",
                ]
                places_raw = [p for p in places_raw if p.get("category_name", "") in allowed_categories]
            
            # 거리 계산 및 필터링
            filtered_places = []
            for p in places_raw:
                try:
                    lat = float(p["y"])
                    lng = float(p["x"])
                    distance_from_start = haversine_km(start_lat, start_lng, lat, lng)
                    error_margin_km = 1.5  # 확대된 오차 범위
                    if abs(distance_from_start - target_distance_from_start) <= error_margin_km:
                        p_copy = dict(p)
                        p_copy["distance_from_start_km"] = distance_from_start
                        p_copy["distance_error"] = abs(distance_from_start - target_distance_from_start)
                        filtered_places.append(p_copy)
                except Exception:
                    continue
            
            if filtered_places:
                filtered_places.sort(key=lambda x: x.get("distance_error", float('inf')))
                places = filtered_places
            else:
                # 여전히 없으면 가장 가까운 것 선택
                places = [dict(p, **{"distance_from_start_km": haversine_km(start_lat, start_lng, float(p["y"]), float(p["x"]))}) for p in places_raw]
                places.sort(key=lambda x: abs(x.get("distance_from_start_km", float('inf')) - target_distance_from_start))
        
        if not places:
            # 여전히 없으면 목표 지점을 그대로 사용
            selected = {
                "place_name": f"{waypoint.theme_keyword} 목적지",
                "address_name": None,
                "road_address_name": None,
                "phone": None,
                "place_url": None,
                "category_name": waypoint.theme_keyword,
                "x": str(target_lng),
                "y": str(target_lat),
            }
        else:
            # 이미 사용된 장소 제외
            available_places = [p for p in places if p.get("place_name") not in used_place_names]
            if not available_places:
                # 사용 가능한 장소가 없으면 전체에서 선택 (최후의 수단)
                available_places = places
            
            # 랜덤으로 선택 (오차가 작은 상위 5개 중에서)
            top_candidates = available_places[:min(5, len(available_places))]
            selected = random.choice(top_candidates) if top_candidates else available_places[0]
            
            # 선택된 장소를 사용된 목록에 추가
            place_name = selected.get("place_name") or f"{waypoint.theme_keyword} 목적지"
            used_place_names.add(place_name)

        total_candidates += len(places)

        # 리뷰 요약 조회
        review_summary = None
        if db:
            review_summary = get_review_summary_for_place(
                db=db,
                place_name=selected.get("place_name") or f"{waypoint.theme_keyword} 목적지",
                place_address=selected.get("address_name") or selected.get("road_address_name"),
                place_lat=float(selected["y"]),
                place_lng=float(selected["x"]),
            )
        
        # 실제 거리 계산
        actual_segment_distance = haversine_km(
            current_lat, current_lng, float(selected["y"]), float(selected["x"])
        )
        accumulated_distance += actual_segment_distance
        
        waypoint_result = WaypointResult(
            place_name=selected.get("place_name")
            or f"{waypoint.theme_keyword} 목적지",
            address_name=selected.get("address_name"),
            road_address_name=selected.get("road_address_name"),
            phone=selected.get("phone"),
            place_url=selected.get("place_url"),
            category_name=selected.get("category_name"),
            x=selected.get("x"),
            y=selected.get("y"),
            distance_km=actual_segment_distance,
            theme_keyword=waypoint.theme_keyword,
            order=waypoint.order,
            review_summary=review_summary,
        )
        waypoint_results.append(waypoint_result)

        route_points.append(
            {
                "name": waypoint_result.place_name,
                "lat": waypoint_result.y,
                "lng": waypoint_result.x,
            }
        )

        current_lat = float(selected["y"])
        current_lng = float(selected["x"])

    if is_round_trip:
        route_points.append(
            {"name": "Start", "lat": f"{start_lat}", "lng": f"{start_lng}"}
        )

    route_url = build_kakao_walk_url(route_points)

    # 실제 총 거리 계산
    actual_total_distance = sum(w.distance_km for w in waypoint_results)

    if is_round_trip and waypoint_results:
        last_waypoint = waypoint_results[-1]
        return_distance = haversine_km(
            float(last_waypoint.y),
            float(last_waypoint.x),
            start_lat,
            start_lng,
        )
        actual_total_distance += return_distance
        accumulated_distance += return_distance

    return RecommendResponse(
        waypoints=waypoint_results,
        route_url=route_url,
        total_distance_km=total_distance_km,
        is_round_trip=is_round_trip,
        candidates_considered=total_candidates,
        actual_total_distance_km=round(actual_total_distance, 2),
    )



