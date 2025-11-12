.PHONY: install run clean

install:
	pip install -r requirements.txt

run:
	PYTHONPATH=./src uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload --env-file .env

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
