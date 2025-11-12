import math
import random
from typing import List, Tuple

class RouteCalculator:
    """
    경로 계산과 관련된 유틸리티 함수를 제공하는 클래스입니다.
    두 지점 간의 거리 계산, 목적지 좌표 계산, 경유지 간 거리 분배 등의 기능을 포함합니다.
    """
    def haversine_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        지구상의 두 지점 간의 대원 거리를 킬로미터 단위로 계산합니다.
        """
        R = 6371.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def calculate_destination_point(self, start_lat: float, start_lng: float, distance_km: float, bearing_deg: float) -> tuple[float, float]:
        """
        시작점에서 특정 거리와 방향으로 떨어진 지점의 좌표를 계산합니다.
        """
        R = 6371.0  # 지구 반지름 (km)
        
        # 각도를 라디안으로 변환
        bearing_rad = math.radians(bearing_deg)
        lat1_rad = math.radians(start_lat)
        lng1_rad = math.radians(start_lng)
        
        # 목적지 좌표 계산
        lat2_rad = math.asin(
            math.sin(lat1_rad) * math.cos(distance_km / R) +
            math.cos(lat1_rad) * math.sin(distance_km / R) * math.cos(bearing_rad)
        )
        
        lng2_rad = lng1_rad + math.atan2(
            math.sin(bearing_rad) * math.sin(distance_km / R) * math.cos(lat1_rad),
            math.cos(distance_km / R) - math.sin(lat1_rad) * math.sin(lat2_rad)
        )
        
        # 라디안을 도로 변환
        dest_lat = math.degrees(lat2_rad)
        dest_lng = math.degrees(lng2_rad)
        
        return dest_lat, dest_lng

    def distribute_distance_for_waypoints(self, total_distance_km: float, waypoint_count: int, is_round_trip: bool) -> List[float]:
        """
        총 거리를 경유지들 사이의 거리로 분배합니다.
        """
        if waypoint_count == 0:
            return []
        
        # 왕복일 경우 마지막에 시작점으로 돌아가는 거리도 고려
        if is_round_trip:
            # 각 구간의 평균 거리 계산
            segment_count = waypoint_count + 1  # 경유지들 사이 + 마지막 경유지에서 시작점으로
            avg_distance = total_distance_km / segment_count
            distances = [avg_distance] * waypoint_count
        else:
            # 편도일 경우 경유지들 사이의 거리만 계산
            avg_distance = total_distance_km / waypoint_count
            distances = [avg_distance] * waypoint_count
        
        return distances
