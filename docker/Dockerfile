FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# System dependencies for building wheels and optional libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

COPY app /app/app
COPY configs /app/configs

EXPOSE 5000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "5000"]
