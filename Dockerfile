FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
RUN pip install .

COPY main.py ./

ENV RAG_DATA_DIR=/data \
    MISTRAL_MODEL=mistral-small-latest \
    MISTRAL_EMBEDDING_MODEL=mistral-embed \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

EXPOSE 7860

CMD ["python", "main.py"]
