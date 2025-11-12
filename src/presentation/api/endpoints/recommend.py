from fastapi import APIRouter, Request, Depends
from slowapi import Limiter
from slowapi.util import get_remote_address

from domain.models import RecommendRequest, RecommendResponse
from infrastructure.kakao.kakao_keyword_search_service import KakaoKeywordSearchService
from infrastructure.services.route_calculator import RouteCalculator
from infrastructure.waypoints.waypoint_finder import WaypointFinder
from infrastructure.waypoints.waypoint_helper import WaypointHelper
from infrastructure.waypoints.strategies.single_destination_strategy import SingleDestinationStrategy
from infrastructure.waypoints.strategies.waypoints_strategy import WaypointsStrategy
from infrastructure.services.recommendation_service import RouteRecommendationService


router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


# 의존성 주입을 위한 헬퍼 함수
def get_kakao_service() -> KakaoKeywordSearchService:
    return KakaoKeywordSearchService()

def get_route_calculator() -> RouteCalculator:
    return RouteCalculator()

def get_waypoint_finder(
    kakao_service: KakaoKeywordSearchService = Depends(get_kakao_service),
    route_calculator: RouteCalculator = Depends(get_route_calculator)
) -> WaypointFinder:
    return WaypointFinder(kakao_service, route_calculator)

def get_waypoint_helper(
    route_calculator: RouteCalculator = Depends(get_route_calculator)
) -> WaypointHelper:
    return WaypointHelper(route_calculator)

def get_recommendation_service(
    req: RecommendRequest,
    kakao_service: KakaoKeywordSearchService = Depends(get_kakao_service),
    route_calculator: RouteCalculator = Depends(get_route_calculator),
    waypoint_finder: WaypointFinder = Depends(get_waypoint_finder),
    waypoint_helper: WaypointHelper = Depends(get_waypoint_helper)
) -> RouteRecommendationService:
    if not req.waypoints:
        strategy = SingleDestinationStrategy(
            kakao_service, route_calculator, waypoint_finder, waypoint_helper
        )
    else:
        strategy = WaypointsStrategy(
            kakao_service, route_calculator, waypoint_finder, waypoint_helper
        )
    return RouteRecommendationService(strategy)


@router.post("/api/recommend", response_model=RecommendResponse)
@limiter.limit("10/minute")
async def recommend(
    request: Request,
    req: RecommendRequest,
    recommendation_service: RouteRecommendationService = Depends(get_recommendation_service)
) -> RecommendResponse:
    return await recommendation_service.recommend_route(req)
