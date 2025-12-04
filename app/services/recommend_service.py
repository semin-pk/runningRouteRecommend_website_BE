import math
import random
from typing import Any, Dict, List, Tuple

import httpx
from fastapi import HTTPException

from app.core.config import KAKAO_REST_API_KEY
from app.schemas.recommend import (
    RecommendRequest,
    RecommendResponse,
    WaypointResult,
)


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
) -> List[Dict[str, Any]]:
    """Search places around a waypoint using Kakao API."""
    places = await kakao_keyword_search(
        keyword,
        x=center_lng,
        y=center_lat,
        radius_m=search_radius_m,
        page_limit=30,
    )

    scored: List[Dict[str, Any]] = []
    for p in places:
        try:
            lat = float(p["y"])
            lng = float(p["x"])
        except Exception:
            continue

        d_km = haversine_km(center_lat, center_lng, lat, lng)
        p_copy = dict(p)
        p_copy["distance_km"] = d_km
        scored.append(p_copy)

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


async def recommend_route(req: RecommendRequest) -> RecommendResponse:
    """Main business logic for route recommendation."""
    start_lat = req.start_lat
    start_lng = req.start_lng
    total_distance_km = req.total_distance_km
    waypoints = req.waypoints
    is_round_trip = req.is_round_trip

    # case: no waypoints
    if not waypoints:
        random_bearing = random.uniform(0, 360)
        target_lat, target_lng = calculate_destination_point(
            start_lat, start_lng, total_distance_km, random_bearing
        )

        places = await kakao_keyword_search(
            "카페",
            x=target_lng,
            y=target_lat,
            radius_m=1000,
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

        selected = places[0]
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

        waypoint_result = WaypointResult(
            place_name=dest_name,
            address_name=selected.get("address_name"),
            road_address_name=selected.get("road_address_name"),
            phone=selected.get("phone"),
            place_url=selected.get("place_url"),
            category_name=selected.get("category_name"),
            x=selected.get("x"),
            y=selected.get("y"),
            distance_km=haversine_km(
                start_lat, start_lng, float(selected["y"]), float(selected["x"])
            ),
            theme_keyword="카페",
            order=1,
        )

        actual_total_distance = waypoint_result.distance_km
        if is_round_trip:
            actual_total_distance += waypoint_result.distance_km

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

    for i, waypoint in enumerate(sorted(waypoints, key=lambda w: w.order)):
        segment_distance = (
            segment_distances[i]
            if i < len(segment_distances)
            else segment_distances[-1]
        )

        random_bearing = random.uniform(0, 360)
        target_lat, target_lng = calculate_destination_point(
            current_lat, current_lng, segment_distance, random_bearing
        )

        places = await find_waypoint_places(
            waypoint.theme_keyword,
            target_lat,
            target_lng,
            1000,
        )

        if not places:
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
            selected = places[0]

        total_candidates += len(places)

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
            distance_km=haversine_km(
                current_lat, current_lng, float(selected["y"]), float(selected["x"])
            ),
            theme_keyword=waypoint.theme_keyword,
            order=waypoint.order,
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

    return RecommendResponse(
        waypoints=waypoint_results,
        route_url=route_url,
        total_distance_km=total_distance_km,
        is_round_trip=is_round_trip,
        candidates_considered=total_candidates,
        actual_total_distance_km=round(actual_total_distance, 2),
    )



