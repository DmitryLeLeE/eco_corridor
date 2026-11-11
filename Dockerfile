FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    ECOCORRIDOR_CACHE=/data/cache \
    PORT=8000

# rasterio wheels link against libexpat, which the slim image lacks
RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install . \
    && useradd --create-home --uid 1000 eco \
    && mkdir -p /data/cache && chown -R eco /data

USER eco
VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:%s/api/health' % os.environ.get('PORT','8000'),timeout=4)"
CMD ["sh", "-c", "exec ecocorridor --host 0.0.0.0 --port ${PORT}"]
