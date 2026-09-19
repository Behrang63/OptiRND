# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /install

COPY requirements.txt .

RUN pip wheel --no-cache-dir --wheel-dir /install/wheels -r requirements.txt


# Stage 2: Runner
FROM python:3.11-slim AS runner

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/home/appuser/.local/bin:$PATH"

RUN groupadd -g 10001 appuser && \
    useradd -u 10001 -g appuser -m -d /home/appuser -s /bin/bash appuser

COPY --from=builder /install/wheels /install/wheels

RUN pip install --no-cache-dir /install/wheels/* && \
    rm -rf /install/wheels

WORKDIR /app

COPY --chown=appuser:appuser . .

USER appuser

EXPOSE 8001

CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8001"]