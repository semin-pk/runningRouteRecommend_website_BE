from typing import Any, Dict, List
from src.domain.models import WaypointResult
from src.infrastructure.services.route_calculator import RouteCalculator
from src.infrastructure.kakao.dto.kakao_keyword_dto import KakaoKeywordSearchDocument


class WaypointHelper:
    """
    경유지 관련 유틸리티 함수를 제공하는 클래스입니다.
    WaypointResult 객체 생성 및 실제 총 거리 계산 기능을 포함합니다.
    """
    def __init__(self, route_calculator: RouteCalculator):
        self.route_calculator = route_calculator

    def build_waypoint_result(
        self,
        selected_place: KakaoKeywordSearchDocument,
        theme_keyword: str,
        order: int,
        start_lat: float,
        start_lng: float
    ) -> WaypointResult:
        """
        주어진 장소 정보와 테마 키워드, 순서, 시작 지점 좌표를 기반으로 WaypointResult 객체를 생성합니다.
        """
        return WaypointResult(
            place_name=selected_place.place_name or f"{theme_keyword} 목적지",
            address_name=selected_place.address_name,
            road_address_name=selected_place.road_address_name,
            phone=selected_place.phone,
            place_url=selected_place.place_url,
            category_name=selected_place.category_name,
            x=selected_place.x,
            y=selected_place.y,
            distance_km=self.route_calculator.haversine_km(
                start_lat, start_lng, float(selected_place.y), float(selected_place.x)
            ),
            theme_keyword=theme_keyword,
            order=order
        )

    def calculate_actual_total_distance(
        self,
        waypoint_results: List[WaypointResult],
        is_round_trip: bool,
        start_lat: float,
        start_lng: float
    ) -> float:
        """
        경유지 목록, 왕복 여부, 시작 지점 좌표를 기반으로 실제 총 거리를 계산합니다.
        """
        actual_total_distance = sum(wr.distance_km for wr in waypoint_results)
        if is_round_trip and waypoint_results:
            last_waypoint = waypoint_results[-1]
            return_distance = self.route_calculator.haversine_km(
                float(last_waypoint.y), float(last_waypoint.x), start_lat, start_lng
            )
            actual_total_distance += return_distance
        return round(actual_total_distance, 2)
