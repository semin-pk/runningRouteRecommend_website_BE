from pydantic import BaseModel
from typing import Optional, List

class KakaoKeywordSearchDocument(BaseModel):
    id: str
    place_name: str
    category_name: str
    category_group_code: Optional[str] = None
    category_group_name: Optional[str] = None
    phone: Optional[str] = None
    address_name: Optional[str] = None
    road_address_name: Optional[str] = None
    x: str
    y: str
    place_url: Optional[str] = None
    distance: Optional[str] = None


class KakaoKeywordSearchResponse(BaseModel):
    documents: List[KakaoKeywordSearchDocument]