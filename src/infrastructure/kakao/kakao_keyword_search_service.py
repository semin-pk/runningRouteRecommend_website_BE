import os
from typing import Any, Dict, List
import httpx
from fastapi import HTTPException

from infrastructure.kakao.dto.kakao_keyword_dto import KakaoKeywordSearchResponse


class KakaoKeywordSearchService:
    """
    카카오 API와 상호작용하는 서비스 클래스입니다.
    키워드 검색 및 카카오 맵 보행 경로 URL 생성을 담당합니다.
    """

    def __init__(self):
        self.KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_API_KEY")
        if not self.KAKAO_REST_API_KEY:
            raise ValueError("KAKAO_REST_API_KEY 환경 변수가 설정되지 않았습니다.")

        self.bearer_type = "KakaoAK"
        self.url = "https://dapi.kakao.com/v2/local/search/keyword"

    async def kakao_keyword_search(
            self,
            query: str,
            x: float,
            y: float,
            radius_m: int,
            page_limit: int = 30,
    ) -> KakaoKeywordSearchResponse:
        """
        주어진 좌표와 반경 내에서 카카오 로컬 키워드 검색 API를 쿼리합니다.
        page_limit까지 여러 페이지를 수집합니다.
        """
        headers = {"Authorization": f"{self.bearer_type} {self.KAKAO_REST_API_KEY}"}

        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                self.url,
                headers=headers,
                params={
                    "query": query,
                    "x": x,
                    "y": y,
                    "radius": radius_m,
                    "sort": "distance",
                    "page": page_limit,
                    "size": 15,
                })

            if r.status_code != 200:
                error_text = r.text
                if "NotAuthorizedError" in error_text and "OPEN_MAP_AND_LOCAL" in error_text:
                    raise HTTPException(
                        status_code=403,
                        detail="Kakao Local API service is not enabled. Please enable 'OPEN_MAP_AND_LOCAL' service in your Kakao Developers console."
                    )
                raise HTTPException(status_code=502, detail=f"Kakao API error: {error_text}")
            data = r.json()

        return KakaoKeywordSearchResponse(**data)

    @staticmethod
    def build_kakao_walk_url(points: List[Dict[str, str]]) -> str:
        """
        /link/by/walk 패턴을 사용하여 카카오 맵 보행 경로 URL을 생성합니다.
        points: 이름, 위도, 경도(문자열) 키를 가진 딕셔너리 목록입니다.
        """
        parts = []
        for p in points:
            name = p.get("name", "Point")
            lat = p["lat"]
            lng = p["lng"]
            parts.append(f"{name},{lat},{lng}")
        return "https://map.kakao.com/link/by/walk/" + "/".join(parts)
