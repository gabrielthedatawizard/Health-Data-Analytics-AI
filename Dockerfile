FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf2.0-0 \
    shared-mime-info \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-mvp.txt ./requirements-mvp.txt
RUN pip install --upgrade pip && pip install -r requirements-mvp.txt

COPY . .

EXPOSE 8000 8501
