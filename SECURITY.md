# 보안 설정 가이드

## 구현된 보안 기능

### 1. CORS 설정
- 프로덕션 환경에서는 `https://www.run2style.com`과 `https://run2style.com`만 허용
- 개발 환경에서는 localhost도 허용
- 서버리스 설정에서도 CORS가 이중으로 설정됨

### 2. API 키 보안
- 하드코딩된 API 키 제거
- AWS SSM Parameter Store를 통한 안전한 키 관리
- 환경변수로만 API 키 관리

### 3. 보안 헤더
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security: max-age=31536000; includeSubDomains`
- `Referrer-Policy: strict-origin-when-cross-origin`

### 4. Rate Limiting
- `/api/recommend` 엔드포인트: 10회/분
- `/health_check` 엔드포인트: 30회/분
- IP 기반 제한

### 5. 입력 검증 강화
- 위도/경도 범위 검증 (-90~90, -180~180)
- 총 거리 제한 (최대 50km)
- 경유지 개수 제한 (최대 10개)
- 키워드 길이 및 특수문자 제한

### 6. Trusted Host 설정
- 허용된 호스트만 접근 가능
- `www.run2style.com`, `run2style.com`, AWS 도메인, localhost

### 7. 에러 핸들링
- 프로덕션 환경에서는 상세한 에러 정보 숨김
- 구조화된 에러 응답

### 8. Lambda 보안 설정
- 동시 실행 제한 (10개)
- Dead Letter Queue 설정
- CloudWatch Logs 보존 기간 설정 (14일)

## 배포 전 확인사항

1. **AWS SSM Parameter Store 설정**
   ```bash
   aws ssm put-parameter \
     --name "/fastapi-running-route/KAKAO_REST_API_KEY" \
     --value "your_actual_api_key" \
     --type "SecureString"
   ```

2. **환경변수 확인**
   - `ENVIRONMENT=production` 설정 확인
   - `KAKAO_REST_API_KEY`가 SSM에서 올바르게 로드되는지 확인

3. **도메인 설정**
   - Amplify 도메인이 `www.run2style.com`으로 설정되어 있는지 확인
   - HTTPS 인증서가 올바르게 설정되어 있는지 확인

4. **모니터링 설정**
   - CloudWatch에서 Lambda 함수 로그 모니터링
   - Rate Limiting 트리거 모니터링
   - Dead Letter Queue 모니터링

## 추가 보안 권장사항

1. **WAF (Web Application Firewall) 설정**
   - AWS WAF를 Lambda 앞에 배치하여 추가 보안 계층 구성

2. **API Gateway 사용**
   - Lambda 직접 호출 대신 API Gateway 사용 고려
   - API 키 관리, 사용량 계획, 스로틀링 등 추가 기능 활용

3. **모니터링 및 알림**
   - CloudWatch 알람 설정
   - 비정상적인 트래픽 패턴 감지

4. **정기적인 보안 점검**
   - 의존성 업데이트
   - 보안 취약점 스캔
   - 로그 분석
