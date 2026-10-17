# Reproducible environment for the pipeline and the Streamlit demo.
#   docker build -t explainable-kg-classification .
#   docker run --rm -p 8501:8501 explainable-kg-classification     # demo
#   docker run --rm explainable-kg-classification python -m src.pipeline
FROM python:3.12-slim

# LightGBM needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Dependencies first, so code changes don't invalidate this layer.
COPY requirements.txt requirements-demo.txt ./
RUN pip install -r requirements-demo.txt

COPY pyproject.toml ./
COPY src ./src
COPY app ./app
COPY .streamlit ./.streamlit
COPY tests ./tests
COPY data ./data
COPY outputs ./outputs

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app/demo.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true"]
