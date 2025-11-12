from src.domain.models import RecommendRequest, RecommendResponse
from src.infrastructure.services.route_recommendation_strategy import RouteRecommendationStrategy


class RouteRecommendationService:
    """
    경로 추천 전략을 사용하여 경로를 추천하는 서비스 클래스입니다.
    다양한 추천 전략을 주입받아 사용할 수 있습니다.
    """
    def __init__(self, strategy: RouteRecommendationStrategy):
        self.strategy = strategy

    async def recommend_route(self, req: RecommendRequest) -> RecommendResponse:
        """
        주어진 요청에 따라 경로를 추천합니다.
        """
        return await self.strategy.recommend_route(req)
