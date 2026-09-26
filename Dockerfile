# Training / evaluation image. PyTorch CUDA wheels are resolved from pyproject.toml.
FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /uvx /bin/

WORKDIR /app
ENV PATH="/app/.venv/bin:${PATH}" \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock README.md .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY configs ./configs
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

ENTRYPOINT ["dl-template"]
CMD ["--help"]
