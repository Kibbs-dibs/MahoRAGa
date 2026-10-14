# ═══════════════════════════════════════════════════════════════════════════
# Project Aletheia — Production Dockerfile
# ═══════════════════════════════════════════════════════════════════════════

FROM python:3.12-slim

# Set environment flags
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    WEB_UI_HOST=0.0.0.0 \
    WEB_UI_PORT=7860

WORKDIR /app

# Install system utilities and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application source code and corpus
COPY core/ core/
COPY graph/ graph/
COPY memory/ memory/
COPY ingestion/ ingestion/
COPY ui/ ui/
COPY RAGxamples/ RAGxamples/
COPY data/ data/
COPY pytest.ini .

# Expose Gradio Web Interface port
EXPOSE 7860

# Default healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Start Gradio interface
CMD ["python", "-m", "ui.app"]
