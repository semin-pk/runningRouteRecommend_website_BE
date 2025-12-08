from pathlib import Path
from dotenv import load_dotenv
import os


load_dotenv()

print("cwd =", os.getcwd())
print("KAKAO_REST_API_KEY =", repr(os.getenv("KAKAO_REST_API_KEY")))
print("OPENAI_API_KEY    =", repr(os.getenv("OPENAI_API_KEY")))
