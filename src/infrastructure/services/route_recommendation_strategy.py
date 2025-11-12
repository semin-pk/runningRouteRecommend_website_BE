from abc import ABC, abstractmethod
from src.domain.models import RecommendRequest, RecommendResponse


class RouteRecommendationStrategy(ABC):
    """
    경로 추천 전략을 정의하는 추상 기본 클래스입니다.
    모든 구체적인 전략은 이 클래스를 상속받아 recommend_route 메서드를 구현해야 합니다.
    """
    @abstractmethod
    async def recommend_route(self, req: RecommendRequest) -> RecommendResponse:
        """
        주어진 요청에 따라 경로를 추천하는 추상 메서드입니다.
        """
        pass
