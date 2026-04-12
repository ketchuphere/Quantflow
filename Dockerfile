FROM python:3.11-slim

LABEL org.opencontainers.image.title="Quantflow — Treasury Cash Position Planner"
LABEL org.opencontainers.image.description="OpenEnv treasury operations simulation"
LABEL org.opencontainers.image.version="1.0.0"
LABEL openenv="true"

WORKDIR /app


ENV PYTHONPATH=/app:/app/src

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy everything
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Runtime env vars
ENV API_BASE_URL="https://router.huggingface.co/v1"
ENV MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
ENV HF_TOKEN=""

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

#stable startup
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]