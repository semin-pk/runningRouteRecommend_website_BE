"""LLM을 사용한 리뷰 요약 유틸리티"""
from typing import List, Dict

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


# --- 2) LLM + 구조화 출력 준비 ---
def get_llm_model():
    """LLM 모델 초기화"""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not configured")
    
    base_model = ChatOpenAI(
        model_name="gpt-5-nano",        # 빠르고 저렴한 모델
        api_key=OPENAI_API_KEY,
        temperature=0.2,
    )
    return base_model.with_structured_output(ReviewExtraction)


# --- 3) 프롬프트 ---
prompt = PromptTemplate.from_template(
    """
너는 한국어 네이버 블로그 리뷰를 분석해서
가게의 대표 메뉴, 분위기, 추천 대상을 키워드로만 뽑는 역할을 한다.

아래 리뷰 텍스트를 보고,
각 항목당 3~4개의 핵심 키워드를 한국어로만 추출해라.

- main_menu: 자주 언급되는 메뉴 이름
- atmosphere: 매장의 분위기/경험/특징
- recommended_for: 어떤 사람이 방문하면 좋을지 (ex. 연인, 친구, 반려견과 함께 등)

반드시 키워드 위주의 짧은 표현만 사용해라.

리뷰 텍스트:
----------------
{text}
----------------
"""
)


# --- 4) LCEL 체인 ---
def get_summarize_chain():
    """요약 체인 생성"""
    model = get_llm_model()
    return prompt | model


# --- 5) 단일 가게용 함수 ---
def extract_review_keywords(input_text: str) -> ReviewExtraction:
    """단일 가게 리뷰 텍스트에서 키워드 추출"""
    if not input_text.strip():
        return ReviewExtraction(main_menu=[], atmosphere=[], recommended_for=[])
    summarize_chain = get_summarize_chain()
    result: ReviewExtraction = summarize_chain.invoke({"text": input_text})
    return result


# --- 6) 여러 가게 batch 처리 ---
def extract_review_keywords_batch(
    store_to_text: Dict[str, str]
) -> Dict[str, ReviewExtraction]:
    """여러 가게 리뷰를 배치로 처리"""
    if not store_to_text:
        return {}
    
    store_names = list(store_to_text.keys())
    inputs = [{"text": store_to_text[name]} for name in store_names]

    summarize_chain = get_summarize_chain()
    results: List[ReviewExtraction] = summarize_chain.batch(inputs)

    store_to_result: Dict[str, ReviewExtraction] = {}
    for name, res in zip(store_names, results):
        if isinstance(res, ReviewExtraction):
            store_to_result[name] = res
        else:
            store_to_result[name] = ReviewExtraction(
                main_menu=[], atmosphere=[], recommended_for=[]
            )

    return store_to_result


