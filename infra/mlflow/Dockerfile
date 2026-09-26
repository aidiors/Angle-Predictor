FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /uvx /bin/

ARG MLFLOW_VERSION=3.16.1
RUN uv pip install --system "mlflow==${MLFLOW_VERSION}"

EXPOSE 5000
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
