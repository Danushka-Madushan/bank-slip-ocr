# Pin to Debian Bookworm slim for reproducible OpenCV/Paddle C++ ABI compatibility
FROM python:3.11-slim-bookworm

# Runtime & memory optimization environment variables:
# - MALLOC_ARENA_MAX=2 prevents glibc memory bloat in multi-threaded C++ (Paddle/OpenCV)
# - OMP_NUM_THREADS=2 prevents CPU thrashing on 1-2 vCPU VPS instances
# - HOME=/app ensures PaddleOCR caches models inside /app/.paddleocr
# - RECEIPT_DB_PATH points SQLite (and its -wal/-shm files) to a mountable volume directory
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MALLOC_ARENA_MAX=2 \
    OMP_NUM_THREADS=2 \
    HOME=/app \
    RECEIPT_DB_PATH=/app/data/receipts.db

# Install minimal system libraries required by OpenCV, Pillow, and PaddlePaddle
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and persistent data directory for SQLite WAL files
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --home-dir /app --shell /sbin/nologin appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app

WORKDIR /app

# Install Python dependencies first to leverage Docker layer caching
COPY --chown=appuser:appuser requirements.txt .
RUN pip3 install -r requirements.txt

# Switch to non-root user before pre-downloading models so cache ownership matches runtime
USER appuser

# Pre-download and bake PaddleOCR English models into the image layer during build.
# This eliminates cold-start downloads and prevents healthcheck timeouts on boot.
RUN python -c "from paddleocr import PaddleOCR; \
    PaddleOCR(use_angle_cls=True, lang='en', show_log=False)" || \
    python -c "from paddleocr import PaddleOCR; \
    PaddleOCR(lang='en')"

# Copy application source code last for fast rebuilds when editing main.py
COPY --chown=appuser:appuser main.py .

# Persistent storage note:
# Railway does not support Dockerfile VOLUME instructions.
# To persist receipts.db, add a Volume in Railway Dashboard -> Service -> Volumes with mount path: /app/data

EXPOSE 8000

# Health check against /health endpoint (supports Railway's dynamic $PORT)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os, urllib.request; port = os.getenv('PORT', '8000'); urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=3)" || exit 1

# Run Uvicorn with dynamic $PORT support for Railway (defaults to 8000 for local runs)
# Using 'exec' ensures Uvicorn receives Unix signals (SIGTERM/SIGINT) directly for graceful shutdown
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --proxy-headers --timeout-keep-alive 15"]
