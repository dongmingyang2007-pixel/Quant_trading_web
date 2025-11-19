# syntax=docker/dockerfile:1
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# collect static in image (whitenoise serves them at runtime)
RUN python manage.py collectstatic --noinput || true

EXPOSE 8000
CMD ["gunicorn", "-c", "gunicorn.conf.py", "quant_trading_site.wsgi:application"]

