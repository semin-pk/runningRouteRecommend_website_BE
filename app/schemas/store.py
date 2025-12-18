from typing import Optional, List, Any
from uuid import UUID
from decimal import Decimal

from pydantic import BaseModel, Field, ConfigDict, field_serializer


class StoreInfoBase(BaseModel):
    """가게 정보 기본 스키마"""

    name: str = Field(..., max_length=255, description="가게 이름")
    address: str = Field(..., max_length=255, description="상세 주소")
    longitude: Decimal = Field(..., decimal_places=7, max_digits=10, description="경도")
    latitude: Decimal = Field(..., decimal_places=7, max_digits=10, description="위도")
    phone: Optional[str] = Field(None, max_length=30, description="전화번호")
    open_time: Optional[str] = Field(None, max_length=10, description="영업 시작 시간 (예: '10:00')")
    close_time: Optional[str] = Field(None, max_length=10, description="영업 종료 시간 (예: '22:00')")


class StoreInfoCreate(StoreInfoBase):
    """가게 정보 생성 스키마"""

    pass


class StoreInfoUpdate(BaseModel):
    """가게 정보 업데이트 스키마"""

    name: Optional[str] = Field(None, max_length=255)
    address: Optional[str] = Field(None, max_length=255)
    longitude: Optional[Decimal] = Field(None, decimal_places=7, max_digits=10)
    latitude: Optional[Decimal] = Field(None, decimal_places=7, max_digits=10)
    phone: Optional[str] = Field(None, max_length=30)
    open_time: Optional[str] = Field(None, max_length=10)
    close_time: Optional[str] = Field(None, max_length=10)


class StoreInfoResponse(StoreInfoBase):
    """가게 정보 응답 스키마"""

    store_id: str = Field(..., description="가게 UUID")

    @field_serializer('longitude', 'latitude')
    def serialize_decimal(self, value: Decimal) -> float:
        """Decimal을 float로 직렬화"""
        return float(value)

    model_config = ConfigDict(from_attributes=True)


class StoreReviewSummaryBase(BaseModel):
    """가게 리뷰 요약 기본 스키마"""

    store_name: str = Field(..., max_length=255, description="가게 이름")
    main_menu: List[Any] = Field(..., description="대표 메뉴 리스트")
    atmosphere: List[Any] = Field(..., description="분위기/경험")
    recommended_for: List[Any] = Field(..., description="추천 대상")


class StoreReviewSummaryCreate(StoreReviewSummaryBase):
    """가게 리뷰 요약 생성 스키마"""

    store_id: str = Field(..., description="가게 UUID")


class StoreReviewSummaryUpdate(BaseModel):
    """가게 리뷰 요약 업데이트 스키마"""

    store_name: Optional[str] = Field(None, max_length=255)
    main_menu: Optional[List[Any]] = None
    atmosphere: Optional[List[Any]] = None
    recommended_for: Optional[List[Any]] = None


class StoreReviewSummaryResponse(StoreReviewSummaryBase):
    """가게 리뷰 요약 응답 스키마"""

    store_id: str = Field(..., description="가게 UUID")

    model_config = ConfigDict(from_attributes=True)


class StoreWithReviewResponse(StoreInfoResponse):
    """가게 정보와 리뷰 요약을 함께 반환하는 스키마"""

    review_summary: Optional[StoreReviewSummaryResponse] = None

    model_config = ConfigDict(from_attributes=True)


class StoreSearchRequest(BaseModel):
    """상세 검색 요청 스키마 (경유지 지원)"""

    themes: List[str] = Field(..., min_items=1, max_items=10, description="검색 테마/키워드 리스트 (경유지별 테마)")
    latitude: float = Field(..., ge=-90, le=90, description="검색 중심 위도")
    longitude: float = Field(..., ge=-180, le=180, description="검색 중심 경도")
    radius_m: int = Field(default=2000, ge=100, le=20000, description="검색 반경 (미터)")
    start_lat: Optional[float] = Field(None, ge=-90, le=90, description="출발지 위도 (경로 계산용)")
    start_lng: Optional[float] = Field(None, ge=-180, le=180, description="출발지 경도 (경로 계산용)")
    total_distance_km: Optional[float] = Field(None, gt=0, le=50, description="총 러닝 거리 (km, 경로 계산용)")
    is_round_trip: bool = Field(default=True, description="왕복 여부")


class StoreCandidateResponse(StoreInfoResponse):
    """가게 후보 응답 스키마 (리뷰 요약 상태 포함)"""

    summary_status: str = Field(..., description="리뷰 요약 상태: 'ready' or 'processing'")
    review_summary: Optional[StoreReviewSummaryResponse] = None

    model_config = ConfigDict(from_attributes=True)


class RouteCandidate(BaseModel):
    """경로 후보 스키마 (경유지 2개 이상일 때)"""
    
    route_id: str = Field(..., description="경로 ID")
    stores: List[StoreCandidateResponse] = Field(..., description="경로에 포함된 가게들 (순서대로)")
    total_distance_km: float = Field(..., description="예상 총 거리 (km)")
    route_url: Optional[str] = Field(None, description="카카오맵 경로 URL")


class StoreSearchResponse(BaseModel):
    """상세 검색 결과 응답 스키마"""

    search_type: str = Field(..., description="검색 타입: 'single' (단일 테마) or 'route' (경로)")
    stores: Optional[List[StoreCandidateResponse]] = Field(None, description="단일 테마일 때 가게 목록 (최대 3개)")
    routes: Optional[List[RouteCandidate]] = Field(None, description="경유지 2개 이상일 때 경로 목록 (최대 3개)")


class StoreConfirmRequest(BaseModel):
    """가게 확정 요청 스키마 (단일 테마일 때)"""

    store_id: str = Field(..., description="선택된 가게 ID")
    start_lat: float = Field(..., ge=-90, le=90, description="출발지 위도")
    start_lng: float = Field(..., ge=-180, le=180, description="출발지 경도")
    total_distance_km: float = Field(..., gt=0, le=50, description="총 러닝 거리 (km)")
    is_round_trip: bool = Field(default=True, description="왕복 여부")


class RouteConfirmRequest(BaseModel):
    """경로 확정 요청 스키마 (경유지 2개 이상일 때)"""

    store_ids: List[str] = Field(..., min_items=2, max_items=10, description="경로에 포함된 가게 ID 리스트 (순서대로)")
    start_lat: float = Field(..., ge=-90, le=90, description="출발지 위도")
    start_lng: float = Field(..., ge=-180, le=180, description="출발지 경도")
    total_distance_km: float = Field(..., gt=0, le=50, description="총 러닝 거리 (km)")
    is_round_trip: bool = Field(default=True, description="왕복 여부")
