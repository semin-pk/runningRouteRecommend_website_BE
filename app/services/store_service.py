"""가게 검색 및 저장 서비스"""
from typing import List, Optional
from uuid import uuid4
from decimal import Decimal

import httpx
from fastapi import HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.core.config import KAKAO_REST_API_KEY
from app.models.store import StoreInfo, StoreReviewSummary
from app.utils.crawling import crawl_one_store_to_text
from app.utils.llm_summarize import extract_review_keywords
from app.schemas.store import StoreInfoCreate


async def kakao_keyword_search(
    query: str,
    x: float,
    y: float,
    radius_m: int,
    page_limit: int = 30,
) -> List[dict]:
    """Query Kakao Local Keyword Search API around a point within radius."""
    if not KAKAO_REST_API_KEY:
        raise HTTPException(
            status_code=500, detail="KAKAO_REST_API_KEY is not configured"
        )

    headers = {"Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"}
    url = "https://dapi.kakao.com/v2/local/search/keyword"

    async with httpx.AsyncClient(timeout=10.0) as client:
        params = {
            "query": query,
            "x": x,
            "y": y,
            "radius": radius_m,
            "sort": "distance",
            "page": page_limit,
            "size": 15,
        }
        r = await client.get(url, headers=headers, params=params)
        if r.status_code != 200:
            error_text = r.text
            if "NotAuthorizedError" in error_text and "OPEN_MAP_AND_LOCAL" in error_text:
                raise HTTPException(
                    status_code=403,
                    detail=(
                        "Kakao Local API service is not enabled. "
                        "Please enable 'OPEN_MAP_AND_LOCAL' service in your Kakao Developers console."
                    ),
                )
            raise HTTPException(status_code=502, detail=f"Kakao API error: {error_text}")
        data = r.json()
        return data.get("documents", [])


def find_or_create_store(
    db: Session,
    kakao_result: dict,
) -> StoreInfo:
    """
    Kakao API 결과로부터 가게 정보를 찾거나 생성.
    이름, 주소, 좌표가 동일하면 기존 가게를 반환하고, 없으면 새로 생성.
    """
    place_name = kakao_result.get("place_name", "")
    address_name = kakao_result.get("address_name", "")
    road_address_name = kakao_result.get("road_address_name", "")
    
    # 주소는 도로명 주소 우선, 없으면 지번 주소 사용
    address = road_address_name if road_address_name else address_name
    if not address:
        address = place_name  # 주소가 없으면 가게 이름 사용
    
    try:
        longitude = Decimal(str(kakao_result.get("x", "0")))
        latitude = Decimal(str(kakao_result.get("y", "0")))
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid coordinates in Kakao API result")
    
    phone = kakao_result.get("phone", "")
    if phone and len(phone) > 30:
        phone = phone[:30]
    
    # 기존 가게 찾기 (이름, 주소, 좌표 기준)
    existing_store = db.query(StoreInfo).filter(
        and_(
            StoreInfo.name == place_name,
            StoreInfo.address == address,
            StoreInfo.longitude == longitude,
            StoreInfo.latitude == latitude,
        )
    ).first()
    
    if existing_store:
        # 기존 가게 정보 업데이트 (전화번호 등)
        if phone and not existing_store.phone:
            existing_store.phone = phone
        db.commit()
        db.refresh(existing_store)
        return existing_store
    
    # 새 가게 생성
    store_id = str(uuid4())
    new_store = StoreInfo(
        store_id=store_id,
        name=place_name,
        address=address,
        longitude=longitude,
        latitude=latitude,
        phone=phone if phone else None,
        open_time=None,
        close_time=None,
    )
    db.add(new_store)
    db.commit()
    db.refresh(new_store)
    return new_store


async def search_stores_by_theme(
    db: Session,
    theme: str,
    latitude: float,
    longitude: float,
    radius_m: int = 2000,
    limit: int = 3,
) -> List[StoreInfo]:
    """
    테마로 가게를 검색하고, 상위 limit개를 DB에 저장/업데이트 후 반환.
    """
    # Kakao API로 검색
    places = await kakao_keyword_search(
        query=theme,
        x=longitude,
        y=latitude,
        radius_m=radius_m,
        page_limit=30,
    )
    
    if not places:
        return []
    
    # 상위 limit개 선택
    selected_places = places[:limit]
    
    # 각 가게를 DB에 저장/업데이트
    stores = []
    for place in selected_places:
        try:
            store = find_or_create_store(db, place)
            stores.append(store)
        except Exception as e:
            # 개별 가게 저장 실패해도 계속 진행
            print(f"Failed to save store {place.get('place_name')}: {e}")
            continue
    
    return stores


def check_review_summary_exists(db: Session, store_id: str) -> bool:
    """가게의 리뷰 요약이 존재하는지 확인"""
    review = db.query(StoreReviewSummary).filter(
        StoreReviewSummary.store_id == store_id
    ).first()
    return review is not None


def process_review_summary_background(
    store_id: str,
    store_name: str,
):
    """
    백그라운드에서 실행될 리뷰 크롤링 및 요약 함수.
    BackgroundTasks에서 호출됨 (동기 함수).
    독립적인 DB 세션을 생성하여 사용.
    """
    from app.database import SessionLocal
    
    db = SessionLocal()
    try:
        # 1. 크롤링
        _, review_text = crawl_one_store_to_text(store_name)
        
        if not review_text.strip():
            print(f"[REVIEW] {store_name}: 크롤링 결과 없음, 빈 요약으로 저장")
            # 빈 요약 저장
            review_summary = StoreReviewSummary(
                store_id=store_id,
                store_name=store_name,
                main_menu=[],
                atmosphere=[],
                recommended_for=[],
            )
            db.add(review_summary)
            db.commit()
            return
        
        # 2. LLM 요약
        extraction = extract_review_keywords(review_text)
        
        # 3. DB에 저장
        review_summary = StoreReviewSummary(
            store_id=store_id,
            store_name=store_name,
            main_menu=extraction.main_menu,
            atmosphere=extraction.atmosphere,
            recommended_for=extraction.recommended_for,
        )
        
        # 기존 요약이 있으면 업데이트, 없으면 생성
        existing = db.query(StoreReviewSummary).filter(
            StoreReviewSummary.store_id == store_id
        ).first()
        
        if existing:
            existing.store_name = store_name
            existing.main_menu = extraction.main_menu
            existing.atmosphere = extraction.atmosphere
            existing.recommended_for = extraction.recommended_for
        else:
            db.add(review_summary)
        
        db.commit()
        print(f"[REVIEW] {store_name}: 리뷰 요약 완료")
        
    except Exception as e:
        print(f"[REVIEW] {store_name}: 리뷰 요약 실패 - {e}")
        db.rollback()
        # 실패해도 빈 요약 저장
        try:
            existing = db.query(StoreReviewSummary).filter(
                StoreReviewSummary.store_id == store_id
            ).first()
            if not existing:
                review_summary = StoreReviewSummary(
                    store_id=store_id,
                    store_name=store_name,
                    main_menu=[],
                    atmosphere=[],
                    recommended_for=[],
                )
                db.add(review_summary)
                db.commit()
        except Exception:
            db.rollback()
    finally:
        db.close()

