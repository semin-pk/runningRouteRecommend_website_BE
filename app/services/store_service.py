"""가게 검색 및 저장 서비스"""
from typing import List, Optional
from uuid import uuid4
from decimal import Decimal

from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.models.store import StoreInfo, StoreReviewSummary
from app.utils.crawling import crawl_one_store_to_text
from app.utils.llm_summarize import extract_review_keywords
from app.schemas.store import StoreInfoCreate


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
    target_distance_km: Optional[float] = None,
    start_lat: Optional[float] = None,
    start_lng: Optional[float] = None,
) -> List[StoreInfo]:
    """
    테마로 가게를 검색하고, 상위 limit개를 DB에 저장/업데이트 후 반환.
    빠른 검색과 동일한 로직 사용: find_waypoint_places를 사용하여 목표 거리 기반 검색.
    """
    from app.services.recommend_service import find_waypoint_places, kakao_keyword_search, haversine_km, calculate_destination_point
    import random
    
    # 목표 거리가 있으면 빠른 검색과 동일한 로직 사용
    if target_distance_km is not None and start_lat is not None and start_lng is not None:
        # 목표 거리만큼 떨어진 지점 계산 (랜덤 방향) - 빠른 검색과 동일
        random_bearing = random.uniform(0, 360)
        target_lat, target_lng = calculate_destination_point(
            start_lat, start_lng, target_distance_km, random_bearing
        )
        
        # 목표 거리 ± 1km 오차 범위 내에서 검색 - 빠른 검색과 동일
        places = await find_waypoint_places(
            keyword=theme,
            center_lat=target_lat,
            center_lng=target_lng,
            search_radius_m=1000,  # 오차 범위 1km = 1000m
            target_distance_km=target_distance_km,
            start_lat=start_lat,
            start_lng=start_lng,
        )
        
        if not places:
            # 필터링된 결과가 없으면 오차 범위를 넓혀서 재시도 - 빠른 검색과 동일
            places = await kakao_keyword_search(
                theme,
                x=target_lng,
                y=target_lat,
                radius_m=2000,  # 오차 범위를 2km로 확대
                page_limit=30,
            )
            
            if not places:
                return []
            
            # 거리 계산 및 필터링 - 빠른 검색과 동일 (카테고리 필터링 없이)
            filtered_places = []
            for p in places:
                try:
                    lat = float(p["y"])
                    lng = float(p["x"])
                    distance_from_start = haversine_km(start_lat, start_lng, lat, lng)
                    error_margin_km = 1.5  # 확대된 오차 범위
                    if abs(distance_from_start - target_distance_km) <= error_margin_km:
                        p_copy = dict(p)
                        p_copy["distance_from_start_km"] = distance_from_start
                        p_copy["distance_error"] = abs(distance_from_start - target_distance_km)
                        filtered_places.append(p_copy)
                except Exception:
                    continue
            
            if filtered_places:
                filtered_places.sort(key=lambda x: x.get("distance_error", float('inf')))
                places = filtered_places
            else:
                # 여전히 없으면 가장 가까운 것 선택 - 빠른 검색과 동일
                places = [dict(p, **{"distance_from_start_km": haversine_km(start_lat, start_lng, float(p["y"]), float(p["x"]))}) for p in places]
                places.sort(key=lambda x: abs(x.get("distance_from_start_km", float('inf')) - target_distance_km))
        
        # 빠른 검색과 동일한 필터링 로직 사용하되, limit개 선택
        # 오차가 작은 순으로 정렬된 places에서 상위 limit개 선택
        selected_places = places[:limit]
    else:
        # 목표 거리가 없으면 기존 로직: 중심점 주변에서 검색
        places = await kakao_keyword_search(
            theme,
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
    store_address: str = None,
    theme: str = None,
):
    """
    백그라운드에서 실행될 리뷰 크롤링 및 요약 함수.
    BackgroundTasks에서 호출됨 (동기 함수).
    독립적인 DB 세션을 생성하여 사용.
    """
    from app.database import SessionLocal
    import re
    
    db = SessionLocal()
    try:
        # 주소에서 시/구 추출하여 검색 쿼리 생성: "{시} {구} {가게이름} {테마}" 형식
        search_query = store_name
        
        # 주소에서 시/구 추출
        city = ""
        district = ""
        
        if store_address:
            # 정규표현식으로 시/도, 구/군 추출
            city_match = re.search(r'([가-힣]+(?:특별시|광역시|시|도))', store_address)
            district_match = re.search(r'([가-힣]+(?:구|군))', store_address)
            
            if city_match:
                city_full = city_match.group(1)
                # "서울특별시" -> "서울", "경기도" -> "경기", "부산광역시" -> "부산"
                city = city_full.replace("특별시", "").replace("광역시", "").replace("시", "").replace("도", "")
            
            if district_match:
                district = district_match.group(1)
        
        # 검색 쿼리 조합: "{시} {구} {가게이름} {테마}"
        query_parts = []
        
        if city:
            query_parts.append(city)
        if district:
            query_parts.append(district)
        if store_name:
            query_parts.append(store_name)
        if theme:
            query_parts.append(theme)
        
        if len(query_parts) > 1:
            search_query = " ".join(query_parts)
        
        # 검색 쿼리 출력 (명확하게 확인 가능하도록)
        print(f"=" * 80)
        print(f"[REVIEW] {store_name}: 검색 쿼리 생성")
        print(f"[REVIEW]   - 원본 가게명: {store_name}")
        print(f"[REVIEW]   - 원본 주소: {store_address}")
        print(f"[REVIEW]   - 추출된 시: {city if city else '(없음)'}")
        print(f"[REVIEW]   - 추출된 구: {district if district else '(없음)'}")
        print(f"[REVIEW]   - 테마: {theme if theme else '(없음)'}")
        print(f"[REVIEW]   - 최종 검색 쿼리: '{search_query}'")
        print(f"=" * 80)
        
        # 1. 크롤링
        _, review_text = crawl_one_store_to_text(search_query)
        
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
        print(f"[REVIEW] {store_name}: LLM 요약 시작")
        extraction = extract_review_keywords(review_text)
        
        print(f"[REVIEW] {store_name}: extraction 결과 확인")
        print(f"[REVIEW] {store_name}: extraction 타입: {type(extraction)}")
        print(f"[REVIEW] {store_name}: main_menu = {extraction.main_menu}")
        print(f"[REVIEW] {store_name}: atmosphere = {extraction.atmosphere}")
        print(f"[REVIEW] {store_name}: recommended_for = {extraction.recommended_for}")
        
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
            print(f"[REVIEW] {store_name}: 기존 요약 업데이트")
            existing.store_name = store_name
            existing.main_menu = extraction.main_menu
            existing.atmosphere = extraction.atmosphere
            existing.recommended_for = extraction.recommended_for
            db.commit()
            db.refresh(existing)
            print(f"[REVIEW] {store_name}: 업데이트 완료 - 저장된 값 확인")
            print(f"[REVIEW] {store_name}: 저장된 main_menu = {existing.main_menu}")
            print(f"[REVIEW] {store_name}: 저장된 atmosphere = {existing.atmosphere}")
            print(f"[REVIEW] {store_name}: 저장된 recommended_for = {existing.recommended_for}")
        else:
            print(f"[REVIEW] {store_name}: 새 요약 생성")
            db.add(review_summary)
            db.commit()
            db.refresh(review_summary)
            print(f"[REVIEW] {store_name}: 생성 완료 - 저장된 값 확인")
            print(f"[REVIEW] {store_name}: 저장된 main_menu = {review_summary.main_menu}")
            print(f"[REVIEW] {store_name}: 저장된 atmosphere = {review_summary.atmosphere}")
            print(f"[REVIEW] {store_name}: 저장된 recommended_for = {review_summary.recommended_for}")
        
        print(f"[REVIEW] {store_name}: 리뷰 요약 완료 및 DB 저장 완료")
        
    except Exception as e:
        import traceback
        print(f"[REVIEW] {store_name}: 리뷰 요약 실패 - {e}")
        print(f"[REVIEW] {store_name}: 트레이스백:\n{traceback.format_exc()}")
        db.rollback()
        # 실패해도 빈 요약 저장
        try:
            existing = db.query(StoreReviewSummary).filter(
                StoreReviewSummary.store_id == store_id
            ).first()
            if not existing:
                print(f"[REVIEW] {store_name}: 실패했지만 빈 요약 저장 시도")
                review_summary = StoreReviewSummary(
                    store_id=store_id,
                    store_name=store_name,
                    main_menu=[],
                    atmosphere=[],
                    recommended_for=[],
                )
                db.add(review_summary)
                db.commit()
                db.refresh(review_summary)
                print(f"[REVIEW] {store_name}: 빈 요약 저장 완료")
        except Exception as e2:
            print(f"[REVIEW] {store_name}: 빈 요약 저장도 실패 - {e2}")
            db.rollback()
    finally:
        db.close()

