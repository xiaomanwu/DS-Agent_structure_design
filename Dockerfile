FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# scipy/numpy/shap wheels are manylinux — no build toolchain needed.
COPY pyproject.toml README.md LICENSE ./
COPY ds_agent ./ds_agent
COPY docs ./docs
COPY examples ./examples
COPY benchmarks ./benchmarks
COPY scripts ./scripts
COPY tests ./tests

RUN pip install --no-cache-dir -e .

# Non-root; artifact dir is a mounted volume in production.
RUN useradd -m agent && mkdir -p /app/artifacts /app/data && chown -R agent /app
USER agent

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health')" || exit 1

CMD ["uvicorn", "ds_agent.web:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--timeout-keep-alive", "75"]
