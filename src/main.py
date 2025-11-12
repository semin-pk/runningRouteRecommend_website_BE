import uvicorn
from mangum import Mangum

from .settings.app_settings import Settings
from .settings.fastapi_setup import FastAPISetup

# FastAPI 앱 설정
settings = Settings()
fastapi_app_setup = FastAPISetup(settings=settings, title="Running Route Recommender")
app = fastapi_app_setup.get_app()

# Uvicorn 실행 (개발용)
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=settings.PORT, reload=True)

# AWS Lambda 배포를 위한 Mangum 핸들러
handler = Mangum(app)
