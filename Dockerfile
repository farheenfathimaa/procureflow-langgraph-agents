FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/home/app/.cache/huggingface

WORKDIR /app

# libgomp is required by CPU PyTorch; curl is not needed because health checks
# use Python's standard library.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only torch first so sentence-transformers does not pull several
# gigabytes of CUDA libraries into the image.
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY app ./app
COPY agents ./agents
COPY rag ./rag
COPY connectors ./connectors
COPY mcp_server ./mcp_server
COPY evals ./evals

RUN pip install ".[embeddings]"

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /data /home/app/.cache/huggingface \
    && chown -R app:app /app /data /home/app/.cache

USER app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
