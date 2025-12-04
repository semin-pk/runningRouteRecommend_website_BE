import os
from typing import List


ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
KAKAO_REST_API_KEY: str | None = os.getenv("KAKAO_REST_API_KEY")


def get_allowed_origins() -> List[str]:
    """Return CORS allowed origins depending on environment."""
    origins = [
        "https://www.run2yourstyle.com",
        "https://run2yourstyle.com",
        "https://main.d1234567890.amplifyapp.com",
    ]

    if ENVIRONMENT != "production":
        origins.extend(
            [
                "http://localhost:3000",
                "http://localhost:3001",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:3001",
            ]
        )
    return origins



