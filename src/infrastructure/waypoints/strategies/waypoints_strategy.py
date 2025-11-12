import random
from typing import Dict, List, Tuple

from src.domain.models import RecommendRequest, RecommendResponse, WaypointResult, Waypoint
from src.infrastructure.kakao.kakao_keyword_search_service import KakaoKeywordSearchService
from src.infrastructure.kakao.dto.kakao_keyword_dto import KakaoKeywordSearchDocument
from src.infrastructure.services.route_calculator import RouteCalculator
from src.infrastructure.waypoints.waypoint_finder import WaypointFinder
from src.infrastructure.waypoints.waypoint_helper import WaypointHelper
from src.infrastructure.services.route_recommendation_strategy import RouteRecommendationStrategy


class WaypointsStrategy(RouteRecommendationStrategy):
    """
    여러 경유지를 포함하는 경로 추천 전략을 구현하는 클래스입니다.
    각 경유지마다 적절한 장소를 검색하고, 전체 경로를 구성하여 추천 경로를 반환합니다.
    """
    def __init__(
        self,
        kakao_service: KakaoKeywordSearchService,
        route_calculator: RouteCalculator,
        waypoint_finder: WaypointFinder,
        waypoint_helper: WaypointHelper
    ):
        self.kakao_service = kakao_service
        self.route_calculator = route_calculator
        self.waypoint_finder = waypoint_finder
        self.waypoint_helper = waypoint_helper

    async def recommend_route(self, req: RecommendRequest) -> RecommendResponse:
        """
        경유지가 있는 경우의 경로를 처리하고 추천 경로 응답을 반환합니다.
        """
        start_lat, start_lng, segment_distances, waypoints, route_points = (
            self._initialize_route_data(req)
        )
        current_lat, current_lng = start_lat, start_lng
        waypoint_results: List[WaypointResult] = []
        total_candidates = 0

        for i, waypoint in enumerate(sorted(waypoints, key=lambda w: w.order)):
            segment_distance = segment_distances[i]
            waypoint_result, current_lat, current_lng, candidates = (
                await self._process_single_waypoint(
                    waypoint, current_lat, current_lng, segment_distance
                )
            )
            waypoint_results.append(waypoint_result)
            route_points.append({
                "name": waypoint_result.place_name,
                "lat": waypoint_result.y, "lng": waypoint_result.x
            })
            total_candidates += candidates

        return self._finalize_route_response(
            req, waypoint_results, route_points, total_candidates, start_lat, start_lng
        )

    def _initialize_route_data(
        self, req: RecommendRequest
    ) -> Tuple[float, float, List[float], List[Waypoint], List[Dict[str, str]]]:
        """
        경로 처리에 필요한 초기 데이터를 설정합니다.
        시작 위도/경도, 구간별 거리, 경유지 목록, 초기 경로 포인트를 반환합니다.
        """
        start_lat, start_lng = req.start_lat, req.start_lng
        waypoint_count = len(req.waypoints)
        segment_distances = self.route_calculator.distribute_distance_for_waypoints(
            req.total_distance_km, waypoint_count, req.is_round_trip
        )
        route_points = [{"name": "Start", "lat": f"{start_lat}", "lng": f"{start_lng}"}]
        return start_lat, start_lng, segment_distances, req.waypoints, route_points

    async def _process_single_waypoint(
        self,
        waypoint: Waypoint,
        current_lat: float,
        current_lng: float,
        segment_distance: float
    ) -> Tuple[WaypointResult, float, float, int]:
        """
        단일 경유지를 처리하고 WaypointResult, 새로운 현재 위도/경도, 검색된 후보 장소 수를 반환합니다.
        """
        random_bearing = random.uniform(0, 360)
        target_lat, target_lng = self.route_calculator.calculate_destination_point(
            current_lat, current_lng, segment_distance, random_bearing
        )

        places: List[KakaoKeywordSearchDocument] = await self.waypoint_finder.find_waypoint_places(
            waypoint.theme_keyword, target_lat, target_lng, 1000
        )

        # 거리 계산 및 정렬
        scored_places = []
        for p in places:
            try:
                lat = float(p.y)
                lng = float(p.x)
            except (ValueError, KeyError):
                continue
            d_km = self.route_calculator.haversine_km(target_lat, target_lng, lat, lng)
            scored_places.append((d_km, p))
        
        scored_places.sort(key=lambda x: x[0])
        sorted_places = [p for d_km, p in scored_places]

        selected: KakaoKeywordSearchDocument
        if sorted_places:
            selected = sorted_places[0]
        else:
            # 장소를 찾지 못한 경우, 기본 KakaoKeywordSearchDocument 객체 생성
            selected = KakaoKeywordSearchDocument(
                id="no_place",
                place_name=f"{waypoint.theme_keyword} 목적지",
                category_name=waypoint.theme_keyword,
                x=str(target_lng),
                y=str(target_lat),
                place_url=""
            )
        
        total_candidates = len(places)

        waypoint_result = self.waypoint_helper.build_waypoint_result(
            selected, waypoint.theme_keyword, waypoint.order, current_lat, current_lng
        )
        new_current_lat = float(selected.y)
        new_current_lng = float(selected.x)
        return waypoint_result, new_current_lat, new_current_lng, total_candidates

    def _finalize_route_response(
        self,
        req: RecommendRequest,
        waypoint_results: List[WaypointResult],
        route_points: List[Dict[str, str]],
        total_candidates: int,
        start_lat: float,
        start_lng: float
    ) -> RecommendResponse:
        """
        최종 RecommendResponse 객체를 구성하고 반환합니다.
        """
        if req.is_round_trip:
            route_points.append({"name": "Start", "lat": f"{start_lat}", "lng": f"{start_lng}"})

        route_url = self.kakao_service.build_kakao_walk_url(route_points)
        actual_total_distance = self.waypoint_helper.calculate_actual_total_distance(
            waypoint_results, req.is_round_trip, start_lat, start_lng
        )

        return RecommendResponse(
            waypoints=waypoint_results, route_url=route_url,
            total_distance_km=req.total_distance_km,
            is_round_trip=req.is_round_trip, candidates_considered=total_candidates,
            actual_total_distance_km=actual_total_distance
        )
