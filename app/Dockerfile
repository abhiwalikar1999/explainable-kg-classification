# Small image for the hosted demo: only the app, its precomputed
# explanations and two figures (the root Dockerfile also runs the pipeline).
#   docker build -f app/Dockerfile -t explainable-kg-demo .
#   docker run --rm -p 8501:8501 explainable-kg-demo
FROM python:3.12-slim
LABEL org.opencontainers.image.source="https://github.com/abhiwalikar1999/explainable-kg-classification"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY app/requirements.txt app/requirements.txt
RUN pip install -r app/requirements.txt

COPY .streamlit ./.streamlit
COPY app/demo.py app/demo.py
COPY outputs/results/demo_explanations.json outputs/results/
COPY outputs/figures/model_comparison.png \
     outputs/figures/family_importance.png outputs/figures/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app/demo.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true"]
