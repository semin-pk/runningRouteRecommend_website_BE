from fastapi import APIRouter, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.get("/health_check")
@limiter.limit("30/minute")
async def health(request: Request) -> dict[str, str]:
    return {"status": "ok"}
