"""LLM을 사용한 리뷰 요약 유틸리티"""
from typing import List, Dict
import json

import os
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from app.core.config import OPENAI_API_KEY
from app.utils.crawling import crawl_stores_in_threads


# --- 1) 구조화된 Output 모델 ---
class ReviewExtraction(BaseModel):
    main_menu: List[str] = Field(
        ...,
        description="가게에서 많이 언급되는 대표 메뉴 키워드들 (예: 소금빵, 아메리카노, 고구마라떼)",
    )
    atmosphere: List[str] = Field(
        ...,
        description="가게 분위기, 경험, 매장 특징 키워드들 (예: 아늑한, 감성적인, 좌석이 넓은)",
    )
    recommended_for: List[str] = Field(
        ...,
        description="어떤 유형의 사람이 방문하면 좋은지 (예: 연인과 함께, 친구와 수다, 반려견과 함께)",
    )


# --- 2) LLM 모델 준비 ---
def get_llm_model():
    """LLM 모델 초기화"""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not configured")
    
    base_model = ChatOpenAI(
        model_name="gpt-4o-mini",        # 빠르고 저렴한 모델
        api_key=OPENAI_API_KEY,
        temperature=0.2,
    )
    return base_model


# --- 3) 프롬프트 ---
prompt = PromptTemplate.from_template(
    """
너는 한국어 네이버 블로그 리뷰를 분석해서
가게의 대표 메뉴, 분위기, 추천 대상을 키워드로만 뽑는 역할을 한다.

아래 리뷰 텍스트를 보고,
각 항목당 3~4개의 핵심 키워드를 한국어로만 추출해라.

- main_menu: 자주 언급되는 메뉴 이름 (리스트)
- atmosphere: 매장의 분위기/경험/특징 (리스트)
- recommended_for: 어떤 사람이 방문하면 좋을지 (리스트, ex. 연인, 친구, 반려견과 함께 등)

반드시 키워드 위주의 짧은 표현만 사용해라.

중요: 반드시 유효한 JSON 형식으로만 응답해야 한다. 다른 설명이나 텍스트는 포함하지 마라.

리뷰 텍스트:
----------------
{text}
----------------

JSON 형식으로 응답:
"""
)


# --- 4) LCEL 체인 (batch용) ---
def get_summarize_chain():
    """요약 체인 생성 (batch 처리용)"""
    model = get_llm_model()
    return prompt | model


# --- 5) 단일 가게용 함수 ---
def extract_review_keywords(input_text: str) -> ReviewExtraction:
    """단일 가게 리뷰 텍스트에서 키워드 추출"""
    if not input_text.strip():
        print("[LLM] 입력 텍스트가 비어있음")
        return ReviewExtraction(main_menu=[], atmosphere=[], recommended_for=[])
    try:
        print(f"[LLM] 리뷰 요약 시작 (텍스트 길이: {len(input_text)})")
        model = get_llm_model()
        prompt_chain = prompt | model
        response = prompt_chain.invoke({"text": input_text})
        
        # 응답이 AIMessage인 경우 content 추출
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = str(response)
        
        print(f"[LLM] LLM 원본 응답: {content}")
        
        # JSON 파싱 시도
        try:
            # JSON 코드 블록 제거 (```json ... ``` 형식)
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            # JSON 파싱
            result_dict = json.loads(content)
            print(f"[LLM] JSON 파싱 성공: {result_dict}")
            
            # ReviewExtraction으로 변환
            extraction = ReviewExtraction(
                main_menu=result_dict.get("main_menu", []),
                atmosphere=result_dict.get("atmosphere", []),
                recommended_for=result_dict.get("recommended_for", [])
            )
            print(f"[LLM] 변환 완료 - main_menu: {extraction.main_menu}, atmosphere: {extraction.atmosphere}, recommended_for: {extraction.recommended_for}")
            return extraction
            
        except json.JSONDecodeError as e:
            print(f"[LLM] JSON 파싱 실패: {e}")
            print(f"[LLM] 파싱 시도한 내용: {content[:200]}...")
            # JSON 파싱 실패 시 빈 객체 반환
            return ReviewExtraction(main_menu=[], atmosphere=[], recommended_for=[])
            
    except Exception as e:
        import traceback
        print(f"[ERROR] extract_review_keywords 실패: {e}")
        print(f"[ERROR] 트레이스백:\n{traceback.format_exc()}")
        return ReviewExtraction(main_menu=[], atmosphere=[], recommended_for=[])


# --- 6) 여러 가게 batch 처리 ---
def extract_review_keywords_batch(
    store_to_text: Dict[str, str]
) -> Dict[str, ReviewExtraction]:
    """여러 가게 리뷰를 배치로 처리"""
    if not store_to_text:
        return {}
    
    # batch 처리 대신 개별 처리 (더 안정적)
    store_to_result: Dict[str, ReviewExtraction] = {}
    for name, text in store_to_text.items():
        try:
            store_to_result[name] = extract_review_keywords(text)
        except Exception as e:
            print(f"[ERROR] {name} 처리 실패: {e}")
            store_to_result[name] = ReviewExtraction(
                main_menu=[], atmosphere=[], recommended_for=[]
            )
    
    return store_to_result


