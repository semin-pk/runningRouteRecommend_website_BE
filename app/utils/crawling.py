"""네이버 블로그 리뷰 크롤링 유틸리티"""
import re
import time
from urllib.parse import quote_plus
from typing import Dict, List

import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from concurrent.futures import ThreadPoolExecutor, as_completed


def extract_korean(text: str) -> str:
    """문자열에서 한글만 추출해서 공백으로 이어붙임."""
    if not text:
        return ""
    result = re.compile("[가-힣]+").findall(text)
    return " ".join(result)


def build_search_url(store_name: str) -> str:
    """가게 이름으로 네이버 블로그 검색 URL 생성."""
    query = quote_plus(store_name)
    url = f"https://search.naver.com/search.naver?ssc=tab.blog.all&sm=tab_jum&query={query}"
    print(f"[CRAWL] 검색 쿼리: '{store_name}'")
    print(f"[CRAWL] 검색 URL: {url}")
    return url


def _create_driver() -> webdriver.Chrome:
    """단일 크롤링용 Chrome 드라이버 생성."""
    chrome_options = webdriver.ChromeOptions()
    # 디버깅할 땐 아래 줄 주석 처리하고 실제 창 보면서 하면 좋음
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    # Chrome 로그 레벨 조정 (USB/Registration 경고 제거)
    chrome_options.add_argument("--log-level=3")  # INFO 레벨 이상만 표시
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    # 불필요한 로그 제거
    import logging
    logging.getLogger('selenium').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    driver = webdriver.Chrome(options=chrome_options)
    return driver


def crawl_naver_blog_reviews(
    store_name: str,
    scroll_count: int = 1,
    max_posts: int = 5,
) -> pd.DataFrame:
    """
    네이버 블로그에서 특정 가게 이름으로 검색한 뒤
    블로그 글 제목, 링크, 내용, 한글만 추출한 내용을 크롤링해서 DataFrame 반환.

    컬럼: titles, links, contents, only_kor_contents
    """
    driver = _create_driver()

    try:
        search_url = build_search_url(store_name)
        driver.get(search_url)
        time.sleep(3)

        body = driver.find_element(By.CSS_SELECTOR, "body")
        for _ in range(scroll_count):
            body.send_keys(Keys.PAGE_DOWN)
            time.sleep(2)

        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        # 클래스 대신 blog.naver.com 도메인 기준으로 링크 찾기
        url_soup = soup.select('a[nocr="1"][href^="https://blog.naver.com/"]')
        '''url_soup = (
            soup.select(
                "div > div.sds-comps-vertical-layout.sds-comps-full-layout.N8JknjVu2Kc8aLavF6ZA > a"
            )
            + soup.select(
                "div > div.sds-comps-vertical-layout.sds-comps-full-layout.MkfloTjOr2Rg4LLlLwVA > a"
            )
            +soup.select(
                'div > div.sds-comps-vertical-layout.sds-comps-full-layout._IJWW1BBVoZsf1hqzdtq > a'
                )
        )'''
        titles: List[str] = []
        links: List[str] = []

        # 필터링할 경로 패턴
        exclude_patterns = [
            "blog.naver.com/MyBlog.naver",
            "section.blog.naver.com",
            "blog.naver.com/blog",
            "/blog/",  # 일반적인 블로그 목록 페이지
        ]

        for t in url_soup:
            title_text = t.get_text().strip()
            link = t.get("href", "").strip()
            if not link:
                continue
            
            # 마이블로그나 불필요한 경로 필터링
            should_exclude = False
            for pattern in exclude_patterns:
                if pattern in link:
                    should_exclude = True
                    break
            
            if should_exclude:
                print(f"[{store_name}] 필터링된 링크: {link}")
                continue
            
            # 실제 블로그 포스트인지 확인
            # blog.naver.com/사용자ID/포스트번호 형식이거나
            # blog.naver.com/사용자ID 형식이면 포함
            if "blog.naver.com/" in link:
                # 상대 경로를 절대 경로로 변환
                if link.startswith("/"):
                    link = "https://blog.naver.com" + link
                elif not link.startswith("http"):
                    link = "https://" + link
                
                # blog.naver.com 이후 경로 확인
                try:
                    url_parts = link.split("blog.naver.com/", 1)[1].split("/")
                    # 사용자ID가 있고, 포스트번호가 있거나 경로가 더 있는 경우
                    if len(url_parts) >= 1 and url_parts[0] and url_parts[0] != "blog":
                        # 유효한 블로그 링크로 간주
                        pass
                    else:
                        # 유효하지 않은 링크
                        print(f"[{store_name}] 유효하지 않은 링크 형식: {link}")
                        continue
                except (IndexError, ValueError):
                    # 링크 파싱 실패
                    print(f"[{store_name}] 링크 파싱 실패: {link}")
                    continue
            
            if len(links) >= max_posts:
                break
                
            print(f"[{store_name}] BLOG:", title_text, link)
            titles.append(title_text)
            links.append(link)

        print(f"[INFO] {store_name} 링크 개수: {len(links)}")

        contents: List[str] = []

        for idx, link in enumerate(links):
            try:
                driver.get(link)
                time.sleep(2)

                # 구버전 블로그 iframe 시도
                try:
                    driver.switch_to.frame("mainFrame")
                except Exception:
                    pass

                text = ""
                # 새 에디터
                try:
                    text = driver.find_element(By.CSS_SELECTOR, "div.se-main-container").text
                except Exception:
                    # 구 에디터
                    try:
                        text = driver.find_element(By.CSS_SELECTOR, "div#postViewArea").text
                    except Exception:
                        text = ""

                contents.append(text)
                print(f"[{store_name}] CONTENT {idx+1}/{len(links)} len={len(text)}")

            except Exception as e:
                print(f"[{store_name}] ERROR {link}: {e}")
                contents.append("")
            finally:
                try:
                    driver.switch_to.default_content()
                except Exception:
                    pass

        only_kor_contents = [extract_korean(c) for c in contents]

        return pd.DataFrame(
            {
                "titles": titles,
                "links": links,
                "contents": contents,
                "only_kor_contents": only_kor_contents,
            }
        )

    finally:
        driver.quit()


def build_review_input_text(
    review_series,
    max_reviews: int = 8,
    max_chars_per_review: int = 1500,
) -> str:
    """
    DataFrame의 only_kor_contents에서
    최대 max_reviews개, 글당 max_chars_per_review까지 잘라 합침.
    """
    texts: List[str] = []
    for raw in review_series[:max_reviews]:
        if not isinstance(raw, str):
            continue
        text = raw.strip()
        if not text:
            continue
        texts.append(text[:max_chars_per_review])
    return "\n\n---\n\n".join(texts)


def crawl_one_store_to_text(store_name: str) -> tuple[str, str]:
    """
    스레드에서 실행될 함수.
    가게 하나 크롤링 + 리뷰 합친 텍스트까지 반환.
    """
    print(f"[MAIN] {store_name} 크롤링 시작")
    df = crawl_naver_blog_reviews(store_name=store_name, scroll_count=1, max_posts=5)

    if df.empty:
        print(f"[MAIN] {store_name}: 크롤링 결과 없음")
        return store_name, ""

    text = build_review_input_text(df["only_kor_contents"])
    print(f"[MAIN] {store_name}: 리뷰 텍스트 길이 = {len(text)}")
    return store_name, text


def crawl_stores_in_threads(
    stores: List[str],
    max_workers: int = 3,
) -> Dict[str, str]:
    """
    여러 가게를 스레드로 병렬 크롤링.
    각 가게마다 Chrome 드라이버 하나씩 생성해서 사용.

    return: {가게이름: 리뷰합친텍스트}
    """
    store_to_text: Dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_store = {
            executor.submit(crawl_one_store_to_text, store): store for store in stores
        }

        for future in as_completed(future_to_store):
            store = future_to_store[future]
            try:
                s_name, text = future.result()
                store_to_text[s_name] = text
            except Exception as e:
                print(f"[MAIN] {store} 쓰레드 예외: {e}")
                store_to_text[store] = ""

    return store_to_text


