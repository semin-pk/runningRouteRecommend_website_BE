from typing import List

from src.infrastructure.kakao.kakao_keyword_search_service import KakaoKeywordSearchService
from src.infrastructure.kakao.dto.kakao_keyword_dto import KakaoKeywordSearchResponse, KakaoKeywordSearchDocument
from src.infrastructure.services.route_calculator import RouteCalculator


class WaypointFinder:
    """
    카카오 서비스를 이용하여 경유지 주변의 장소를 검색하고,
    경로 계산기를 이용하여 검색된 장소의 거리를 계산하는 클래스입니다.
    """

    def __init__(self, kakao_service: KakaoKeywordSearchService, route_calculator: RouteCalculator):
        self.kakao_service = kakao_service
        self.route_calculator = route_calculator

    async def find_waypoint_places(
            self,
            keyword: str,
            center_lat: float,
            center_lng: float,
            search_radius_m: int = 2000
    ) -> List[KakaoKeywordSearchDocument]:
        """
        경유지 주변에서 키워드로 장소를 검색하고, 각 장소의 거리를 계산하여 반환합니다.
        """
        places: KakaoKeywordSearchResponse = await self.kakao_service.kakao_keyword_search(
            keyword, x=center_lng, y=center_lat, radius_m=search_radius_m, page_limit=30
        )
        return places.documents
