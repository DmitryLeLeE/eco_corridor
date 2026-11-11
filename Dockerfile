FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    ECOCORRIDOR_CACHE=/data/cache

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .

VOLUME /data
EXPOSE 8000
CMD ["ecocorridor", "--host", "0.0.0.0", "--port", "8000"]
