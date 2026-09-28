# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /install

COPY requirements.txt .

RUN pip wheel --no-cache-dir --wheel-dir /install/wheels -r requirements.txt


# Stage 2: Runner
FROM python:3.11-slim AS runner

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app"

RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -M -d /app -s /usr/sbin/nologin appuser

COPY --from=builder /install/wheels /install/wheels

RUN pip install --no-cache-dir /install/wheels/* && \
    rm -rf /install/wheels

RUN mkdir -p /app/data && chown -R appuser:appgroup /app/data

WORKDIR /app

COPY --chown=appuser:appgroup api/ ./api/
COPY --chown=appuser:appgroup core/ ./core/
COPY --chown=appuser:appgroup templates/ ./templates/
COPY --chown=appuser:appgroup config/ ./config/
COPY --chown=appuser:appgroup migrations/ ./migrations/
COPY --chown=appuser:appgroup data/ ./data/

USER appuser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8001/api/v1/health').getcode()==200 else 1)"

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8001"]