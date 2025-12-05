"""데이터베이스 초기화 스크립트"""
import sys
import os

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 모든 모델을 import하여 테이블 정의가 로드되도록 함
from app.models import StoreInfo, StoreReviewSummary  # noqa: F401

from app.database import init_db
from app.core.config import DB_NAME, DB_HOST, DB_PORT

def main():
    """데이터베이스 테이블 생성"""
    print(f"데이터베이스 초기화 시작...")
    print(f"데이터베이스: {DB_NAME}")
    print(f"호스트: {DB_HOST}:{DB_PORT}")
    
    try:
        init_db()
        print("✅ 데이터베이스 테이블 생성 완료!")
        print("\n생성된 테이블:")
        print("  - store_info (가게 정보)")
        print("  - store_review_summary (리뷰 요약)")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        print("\n확인 사항:")
        print("  1. MySQL 서버가 실행 중인지 확인")
        print("  2. .env 파일의 데이터베이스 정보가 올바른지 확인")
        print("  3. 데이터베이스가 생성되었는지 확인")
        sys.exit(1)

if __name__ == "__main__":
    main()

