"""Thin wrapper to keep backward compatibility.

The real application code now lives in `app/main.py` within the `app` package.
"""

import os

from app.main import app, handler  # noqa: F401


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )