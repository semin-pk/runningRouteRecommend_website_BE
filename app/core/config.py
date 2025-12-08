import os
from typing import List
from dotenv import load_dotenv

load_dotenv()
ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
KAKAO_REST_API_KEY: str | None = os.getenv("KAKAO_REST_API_KEY")
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")

# Database configuration
DB_HOST: str = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT: int = int(os.getenv("DB_PORT", "3306"))
DB_USER: str = os.getenv("DB_USER", "semin")
DB_PASSWORD: str = os.getenv("DB_PASSWORD", "semin0809")
DB_NAME: str = os.getenv("DB_NAME", "run2yourstyle_DB")

# SQLAlchemy database URL
DATABASE_URL: str = (
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    f"?charset=utf8mb4"
)


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
                "http://localhost:5173",
                "http://127.0.0.1:3000",
                "http://127.0.0.1:3001",
                "http://127.0.0.1:5173",
            ]
        )
    return origins



