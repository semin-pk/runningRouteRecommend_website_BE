# Running Route Recommender API 명세서

## 개요

러닝 코스를 추천하는 REST API 서비스입니다. 시작 지점과 목표 거리, 경유지 테마를 입력받아 카카오 맵 API를 활용하여 최적의 러닝 경로를 추천합니다.

**Base URL**: `https://api.run2yourstyle.com` (프로덕션)

**버전**: 1.0.0

**인증**: KAKAO_REST_API_KEY (서버 측 환경 변수)

---

## 공통 사항

### 요청 형식
- Content-Type: `application/json`
- Accept: `application/json`

### 응답 형식
- Content-Type: `application/json`
- 모든 응답은 JSON 형식으로 반환됩니다.

### 에러 처리

모든 에러는 다음 형식으로 반환됩니다:

```json
{
  "error": "에러 메시지",
  "status_code": 400
}
```

### Rate Limiting

| 엔드포인트 | 제한 |
|-----------|------|
| `GET /health_check` | 30 requests/minute |
| `POST /api/recommend` | 10 requests/minute |

Rate limit을 초과하면 `429 Too Many Requests` 에러가 반환됩니다.

### 보안 헤더

모든 응답에 다음 보안 헤더가 포함됩니다:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `X-XSS-Protection: 1; mode=block`
- `Strict-Transport-Security: max-age=31536000; includeSubDomains`
- `Referrer-Policy: strict-origin-when-cross-origin`

---

## API 엔드포인트

### 1. 헬스 체크

서버 상태를 확인하는 엔드포인트입니다.

**엔드포인트**: `GET /health_check`

**Rate Limit**: 30 requests/minute

**요청**
- 메서드: `GET`
- 파라미터: 없음

**응답**

**성공 (200 OK)**
```json
{
  "status": "ok"
}
```

**예시**
```bash
curl -X GET https://api.run2yourstyle.com/health_check
```

---

### 2. 러닝 코스 추천

시작 지점, 목표 거리, 경유지 정보를 받아 러닝 코스를 추천합니다.

**엔드포인트**: `POST /api/recommend`

**Rate Limit**: 10 requests/minute

**요청**

**Request Body**

| 필드명 | 타입 | 필수 | 설명 | 제약 조건 |
|--------|------|------|------|----------|
| `start_lat` | float | ✅ | 시작 지점의 위도 | -90 ~ 90 |
| `start_lng` | float | ✅ | 시작 지점의 경도 | -180 ~ 180 |
| `total_distance_km` | float | ✅ | 목표 러닝 거리 (km) | 0 초과, 최대 50 |
| `waypoints` | array | ❌ | 경유지 목록 | 최대 10개 |
| `is_round_trip` | boolean | ❌ | 왕복 여부 | 기본값: true |

**Waypoint 객체**

| 필드명 | 타입 | 필수 | 설명 | 제약 조건 |
|--------|------|------|------|----------|
| `theme_keyword` | string | ✅ | 경유지 검색 키워드 | 1-50자, 특수문자 제한 |
| `order` | integer | ✅ | 경유지 순서 | 1-10, 중복 불가 |

**요청 예시**

**경유지 없이 요청 (기본 카페 추천)**
```json
{
  "start_lat": 37.5665,
  "start_lng": 126.9780,
  "total_distance_km": 5.0,
  "is_round_trip": true
}
```

**경유지 포함 요청**
```json
{
  "start_lat": 37.5665,
  "start_lng": 126.9780,
  "total_distance_km": 7.0,
  "is_round_trip": true,
  "waypoints": [
    {
      "theme_keyword": "카페",
      "order": 1
    },
    {
      "theme_keyword": "맛집",
      "order": 2
    },
    {
      "theme_keyword": "맥주",
      "order": 3
    }
  ]
}
```

**응답**

**성공 (200 OK)**

| 필드명 | 타입 | 설명 |
|--------|------|------|
| `waypoints` | array | 추천된 경유지 목록 |
| `route_url` | string | 카카오 맵 걷기 길찾기 URL |
| `total_distance_km` | float | 목표 러닝 거리 (km) |
| `actual_total_distance_km` | float | 실제 총 거리 (km) |
| `is_round_trip` | boolean | 왕복 여부 |
| `candidates_considered` | integer | 검토된 장소 후보 개수 |

**WaypointResult 객체**

| 필드명 | 타입 | 설명 |
|--------|------|------|
| `place_name` | string | 장소명 |
| `address_name` | string \| null | 지번 주소 |
| `road_address_name` | string \| null | 도로명 주소 |
| `phone` | string \| null | 전화번호 |
| `place_url` | string \| null | 카카오 장소 상세 URL |
| `category_name` | string \| null | 카테고리명 |
| `x` | string | 경도 |
| `y` | string | 위도 |
| `distance_km` | float | 시작점 또는 이전 경유지로부터의 거리 (km) |
| `theme_keyword` | string | 검색에 사용된 키워드 |
| `order` | integer | 경유지 순서 |

**응답 예시**

```json
{
  "waypoints": [
    {
      "place_name": "스타벅스 강남점",
      "address_name": "서울특별시 강남구 역삼동 123",
      "road_address_name": "서울특별시 강남구 테헤란로 456",
      "phone": "02-1234-5678",
      "place_url": "https://place.map.kakao.com/12345678",
      "category_name": "카페",
      "x": "127.0284",
      "y": "37.4985",
      "distance_km": 2.3,
      "theme_keyword": "카페",
      "order": 1
    },
    {
      "place_name": "맛있는 식당",
      "address_name": "서울특별시 강남구 역삼동 789",
      "road_address_name": "서울특별시 강남구 테헤란로 321",
      "phone": "02-9876-5432",
      "place_url": "https://place.map.kakao.com/87654321",
      "category_name": "음식점",
      "x": "127.0334",
      "y": "37.5035",
      "distance_km": 1.8,
      "theme_keyword": "맛집",
      "order": 2
    }
  ],
  "route_url": "https://map.kakao.com/link/by/walk/Start,37.5665,126.9780/스타벅스 강남점,37.4985,127.0284/맛있는 식당,37.5035,127.0334/Start,37.5665,126.9780",
  "total_distance_km": 7.0,
  "actual_total_distance_km": 8.2,
  "is_round_trip": true,
  "candidates_considered": 45
}
```

**에러 응답**

**400 Bad Request** - 잘못된 요청
```json
{
  "error": "Waypoint orders must be unique",
  "status_code": 400
}
```

**403 Forbidden** - 카카오 API 서비스 미활성화
```json
{
  "error": "Kakao Local API service is not enabled. Please enable 'OPEN_MAP_AND_LOCAL' service in your Kakao Developers console.",
  "status_code": 403
}
```

**429 Too Many Requests** - Rate Limit 초과
```json
{
  "error": "Rate limit exceeded",
  "status_code": 429
}
```

**500 Internal Server Error** - 서버 내부 에러
```json
{
  "error": "Internal server error",
  "status_code": 500
}
```

**502 Bad Gateway** - 카카오 API 에러
```json
{
  "error": "Kakao API error: ...",
  "status_code": 502
}
```

**예시 요청 (cURL)**
```bash
curl -X POST https://api.run2yourstyle.com/api/recommend \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{
    "start_lat": 37.5665,
    "start_lng": 126.9780,
    "total_distance_km": 5.0,
    "is_round_trip": true,
    "waypoints": [
      {
        "theme_keyword": "카페",
        "order": 1
      }
    ]
  }'
```

---

## 데이터 모델

### Waypoint
```typescript
{
  theme_keyword: string;  // 1-50자, 특수문자 제한
  order: number;         // 1-10, 고유값
}
```

### RecommendRequest
```typescript
{
  start_lat: number;              // -90 ~ 90
  start_lng: number;              // -180 ~ 180
  total_distance_km: number;      // 0 초과, 최대 50
  waypoints?: Waypoint[];         // 최대 10개
  is_round_trip?: boolean;        // 기본값: true
}
```

### WaypointResult
```typescript
{
  place_name: string;
  address_name: string | null;
  road_address_name: string | null;
  phone: string | null;
  place_url: string | null;
  category_name: string | null;
  x: string;              // 경도
  y: string;              // 위도
  distance_km: number;
  theme_keyword: string;
  order: number;
}
```

### RecommendResponse
```typescript
{
  waypoints: WaypointResult[];
  route_url: string;
  total_distance_km: number;
  actual_total_distance_km: number;
  is_round_trip: boolean;
  candidates_considered: number;
}
```

---

## 동작 원리

### 경유지 없는 경우
- 시작 지점에서 랜덤 방향으로 목표 거리만큼 떨어진 지점을 계산
- 해당 지점 주변 1km 반경에서 "카페" 키워드로 장소 검색
- 가장 가까운 장소를 선택하여 경로 생성

### 경유지 있는 경우
1. 총 거리를 경유지 개수에 따라 균등 분배
   - 왕복: 각 구간 = 총 거리 / (경유지 개수 + 1)
   - 편도: 각 구간 = 총 거리 / 경유지 개수
2. 각 경유지마다:
   - 현재 위치에서 해당 구간 거리만큼 랜덤 방향으로 이동
   - 이동한 지점 주변 1km 반경에서 키워드로 장소 검색
   - 가장 가까운 장소를 선택하여 경유지로 설정
   - 다음 경유지를 위해 현재 위치 업데이트
3. 왕복인 경우 마지막 경유지에서 시작점까지의 거리 추가 계산

### 거리 계산
- 모든 거리 계산은 Haversine 공식을 사용하여 지구 곡률을 고려한 실제 거리를 계산합니다.

---

## 제한 사항

1. **최대 거리**: 50km 이하
2. **최대 경유지**: 10개
3. **검색 반경**: 각 경유지 주변 1-2km
4. **Rate Limit**: 
   - 헬스 체크: 30 requests/minute
   - 코스 추천: 10 requests/minute
5. **키워드 제한**: 특수문자 (`<`, `>`, `&`, `"`, `'`, `\`, `/`) 포함 불가

---

## 외부 의존성

### 카카오 API
- **Kakao Local API** - 키워드로 장소 검색
- 필수 서비스 활성화: `OPEN_MAP_AND_LOCAL`
- API 키는 서버 측 환경 변수 `KAKAO_REST_API_KEY`로 설정 필요

---

## CORS 설정

다음 도메인에서의 요청이 허용됩니다:
- `https://www.run2yourstyle.com`
- `https://run2yourstyle.com`
- 개발 환경: `http://localhost:3000`, `http://localhost:3001`

---

## 변경 이력

### v1.0.0 (현재)
- 초기 릴리스
- 헬스 체크 엔드포인트 추가
- 러닝 코스 추천 엔드포인트 추가
- 경유지 지원
- 왕복/편도 옵션 지원
- Rate Limiting 적용
- 보안 헤더 추가

