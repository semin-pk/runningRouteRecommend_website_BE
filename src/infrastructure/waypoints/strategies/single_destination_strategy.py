import random
from typing import Any, Dict, List, Tuple

from src.domain.models import RecommendRequest, RecommendResponse, WaypointResult
from infrastructure.kakao.kakao_keyword_search_service import KakaoKeywordSearchService
from infrastructure.kakao.dto.kakao_keyword_dto import KakaoKeywordSearchResponse, KakaoKeywordSearchDocument
from src.infrastructure.services.route_calculator import RouteCalculator
from src.infrastructure.waypoints.waypoint_finder import WaypointFinder
from src.infrastructure.waypoints.waypoint_helper import WaypointHelper
from src.infrastructure.services.route_recommendation_strategy import RouteRecommendationStrategy


class SingleDestinationStrategy(RouteRecommendationStrategy):
    """
    단일 목적지 경로 추천 전략을 구현하는 클래스입니다.
    경유지가 없는 경우, 시작점에서 특정 거리와 방향으로 떨어진 지점을 목적지로 설정하고
    해당 목적지 주변의 장소를 검색하여 경로를 추천합니다.
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
        경유지가 없는 경우의 경로를 처리하고 추천 경로 응답을 반환합니다.
        """
        start_lat, start_lng = req.start_lat, req.start_lng
        total_distance_km, is_round_trip = req.total_distance_km, req.is_round_trip

        target_lat, target_lng = self._calculate_target_point(
            start_lat, start_lng, total_distance_km
        )
        places = await self._find_places(target_lat, target_lng)

        if not places:
            return self._handle_no_places_found(start_lat, start_lng, total_distance_km, is_round_trip)

        selected = places[0]
        waypoint_result = self.waypoint_helper.build_waypoint_result(
            selected, "카페", 1, start_lat, start_lng
        )

        route_points = self._build_route_points(start_lat, start_lng, waypoint_result, is_round_trip)
        route_url = self.kakao_service.build_kakao_walk_url(route_points)
        actual_total_distance = self.waypoint_helper.calculate_actual_total_distance(
            [waypoint_result], is_round_trip, start_lat, start_lng
        )

        return RecommendResponse(
            waypoints=[waypoint_result], route_url=route_url,
            total_distance_km=total_distance_km,
            actual_total_distance_km=actual_total_distance,
            is_round_trip=is_round_trip, candidates_considered=len(places)
        )

    def _calculate_target_point(
        self, start_lat: float, start_lng: float, total_distance_km: float
    ) -> Tuple[float, float]:
        """
        시작점에서 주어진 총 거리를 기반으로 임의의 방향으로 떨어진 목표 지점의 위도와 경도를 계산합니다.
        """
        random_bearing = random.uniform(0, 360)
        return self.route_calculator.calculate_destination_point(
            start_lat, start_lng, total_distance_km, random_bearing
        )

    async def _find_places(
        self, target_lat: float, target_lng: float
    ) -> List[KakaoKeywordSearchDocument]:
        """
        목표 지점 주변에서 "카페" 키워드로 장소를 검색하고, 거리를 계산하여 정렬합니다.
        """
        places: List[KakaoKeywordSearchDocument] = await self.waypoint_finder.find_waypoint_places(
            "카페", target_lat, target_lng, 1000
        )

        scored_places = []
        for p in places:
            try:
                lat = float(p.y)
                lng = float(p.x)
            except (ValueError, KeyError):
                continue

            d_km = self.route_calculator.haversine_km(target_lat, target_lng, lat, lng)
            # KakaoKeywordSearchDocument 객체에 distance_km 속성을 추가할 수 없으므로,
            # 임시적으로 딕셔너리로 변환하여 사용하거나, 새로운 DTO를 정의해야 합니다.
            # 여기서는 KakaoKeywordSearchDocument 객체 자체를 반환하고,
            # distance_km는 WaypointResult 생성 시점에 계산하도록 합니다.
            # 따라서 정렬은 하지 않고, 첫 번째 요소를 선택하는 로직은 유지합니다.
            scored_places.append((d_km, p)) # (distance, KakaoKeywordSearchDocument) 튜플로 저장

        scored_places.sort(key=lambda x: x[0]) # 거리를 기준으로 정렬
        return [p for d_km, p in scored_places] # 정렬된 KakaoKeywordSearchDocument 리스트 반환

    def _handle_no_places_found(
        self, start_lat: float, start_lng: float, total_distance_km: float, is_round_trip: bool
    ) -> RecommendResponse:
        """
        주변 장소를 찾지 못한 경우, 빈 경유지 목록과 시작 지점만 포함된 경로 URL로 응답을 생성합니다.
        """
        route_url = self.kakao_service.build_kakao_walk_url(
            [{"name": "Start", "lat": f"{start_lat}", "lng": f"{start_lng}"}]
        )
        return RecommendResponse(
            waypoints=[], route_url=route_url, total_distance_km=total_distance_km,
            actual_total_distance_km=0, is_round_trip=is_round_trip, candidates_considered=0
        )

    def _build_route_points(
        self, start_lat: float, start_lng: float, waypoint_result: WaypointResult, is_round_trip: bool
    ) -> List[Dict[str, str]]:
        """
        시작 지점, 경유지 결과, 왕복 여부를 기반으로 카카오 맵 URL 생성을 위한 경로 포인트를 구성합니다.
        """
        route_points = [
            {"name": "Start", "lat": f"{start_lat}", "lng": f"{start_lng}"},
            {"name": waypoint_result.place_name, "lat": waypoint_result.y, "lng": waypoint_result.x}
        ]
        if is_round_trip:
            route_points.append({"name": "Start", "lat": f"{start_lat}", "lng": f"{start_lng}"})
        return route_points
