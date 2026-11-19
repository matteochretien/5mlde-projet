# Image unique pour l'API, les flows Prefect et la CLI : même code, mêmes
# dépendances verrouillées, donc exactement le même comportement qu'en local.
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates git \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app

# Couche de dépendances séparée du code : un changement de source ne
# réinstalle pas tout l'environnement.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --all-extras

COPY src/ ./src/
COPY api/ ./api/
COPY scripts/ ./scripts/
COPY config/ ./config/
RUN uv sync --frozen --all-extras

EXPOSE 8001

# Par défaut : l'API. Les autres services surchargent `command`.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
