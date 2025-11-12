from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    ENVIRONMENT: str = "development"
    ALLOWED_HOSTS: List[str] = [
        "www.run2yourstyle.com",
        "run2yourstyle.com",
        "*.amazonaws.com",
        "localhost",
        "127.0.0.1",
        "0.0.0.0"
    ]
    ALLOWED_ORIGINS: List[str] = [
        "https://www.run2yourstyle.com",
        "https://run2yourstyle.com",
        "https://main.d1234567890.amplifyapp.com",
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001"
    ]
    PORT: int = 8000

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
