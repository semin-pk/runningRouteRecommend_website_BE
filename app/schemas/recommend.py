from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class Waypoint(BaseModel):
    theme_keyword: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Search keyword for this waypoint",
    )
    order: int = Field(
        ...,
        ge=1,
        le=10,
        description="Order of this waypoint in the route",
    )

    @validator("theme_keyword")
    def strip_keyword(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Theme keyword cannot be empty")
        if any(char in v for char in ['<', '>', '&', '"', "'", "\\", "/"]):
            raise ValueError("Theme keyword contains invalid characters")
        return v


class RecommendRequest(BaseModel):
    start_lat: float = Field(..., ge=-90, le=90, description="Starting latitude")
    start_lng: float = Field(..., ge=-180, le=180, description="Starting longitude")
    total_distance_km: float = Field(
        ...,
        gt=0,
        le=50,
        description="Total desired running distance in kilometers (max 50km)",
    )
    waypoints: List[Waypoint] = Field(
        default=[],
        max_items=10,
        description="List of waypoints to visit (max 10)",
    )
    is_round_trip: bool = Field(
        default=True,
        description="Whether to return to start point (round trip) or end at last waypoint (one way)",
    )

    @validator("waypoints")
    def validate_waypoint_orders(cls, v):
        if v:
            orders = [w.order for w in v]
            if len(set(orders)) != len(orders):
                raise ValueError("Waypoint orders must be unique")
            if min(orders) < 1:
                raise ValueError("Waypoint orders must be >= 1")
        return v


class WaypointResult(BaseModel):
    place_name: str
    address_name: Optional[str]
    road_address_name: Optional[str]
    phone: Optional[str]
    place_url: Optional[str]
    category_name: Optional[str]
    x: str
    y: str
    distance_km: float
    theme_keyword: str
    order: int
    review_summary: Optional[Dict[str, Any]] = Field(None, description="리뷰 요약 정보 (main_menu, atmosphere, recommended_for)")


class RecommendResponse(BaseModel):
    waypoints: List[WaypointResult]
    route_url: str
    total_distance_km: float
    actual_total_distance_km: float
    is_round_trip: bool
    candidates_considered: int



