# 백엔드 환경 변수 설정 가이드

## .env 파일 생성

프로젝트 루트 디렉토리(`runningRouteRecommend_website_BE/`)에 `.env` 파일을 생성하세요.

## 필수 환경 변수

```env
# 환경 설정
ENVIRONMENT=development

# Kakao API Key (REST API 키)
KAKAO_REST_API_KEY=your_kakao_rest_api_key_here

# OpenAI API Key (리뷰 요약용)
OPENAI_API_KEY=your_openai_api_key_here

# 데이터베이스 설정
DB_HOST=127.0.0.1
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password_here
DB_NAME=run2yourstyle_DB

# 서버 포트 (선택사항, 기본값: 8000)
PORT=8000
```

## 환경 변수 설명

- **ENVIRONMENT**: `development` 또는 `production`
- **KAKAO_REST_API_KEY**: 카카오 개발자 콘솔에서 발급받은 REST API 키
- **OPENAI_API_KEY**: OpenAI Platform에서 발급받은 API 키
- **DB_HOST**: MySQL 호스트 주소 (기본값: 127.0.0.1)
- **DB_PORT**: MySQL 포트 (기본값: 3306)
- **DB_USER**: MySQL 사용자명 (기본값: root)
- **DB_PASSWORD**: MySQL 비밀번호
- **DB_NAME**: 데이터베이스 이름 (기본값: run2yourstyle_DB)
- **PORT**: 백엔드 서버 포트 (기본값: 8000)

## Windows에서 .env 파일 생성

1. 메모장 열기
2. 위의 내용 복사하여 붙여넣기
3. 실제 값으로 수정
4. 파일 이름을 `.env`로 저장 (파일 형식: 모든 파일)
5. `runningRouteRecommend_website_BE` 폴더에 저장

## macOS/Linux에서 .env 파일 생성

```bash
cd runningRouteRecommend_website_BE
nano .env
# 또는
vim .env
```

위의 내용을 복사하여 붙여넣고, 실제 값으로 수정한 후 저장하세요.

## 보안 주의사항

- `.env` 파일은 절대 Git에 커밋하지 마세요
- `.gitignore`에 `.env`가 포함되어 있는지 확인하세요
- 프로덕션 환경에서는 환경 변수를 안전하게 관리하세요

