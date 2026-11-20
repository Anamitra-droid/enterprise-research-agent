FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/app/var/hf

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /usr/local/bin/uv

WORKDIR /app

# CPU-only torch keeps the image small (sentence-transformers needs torch).
COPY pyproject.toml uv.lock* README.md ./
RUN uv pip install --system --index-url https://download.pytorch.org/whl/cpu torch \
 && uv pip install --system -r pyproject.toml --extra local-embeddings

COPY src ./src
COPY data ./data
COPY eval ./eval
COPY scripts ./scripts
RUN uv pip install --system --no-deps -e .

# Pre-download the embedding model and build the index at image build time.
RUN python -m research_agent.cli ingest --reset

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "research_agent.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
