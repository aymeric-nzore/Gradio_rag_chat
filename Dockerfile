FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY docker-requirements.txt README.md ./
RUN pip install -r docker-requirements.txt

COPY main.py ./

ENV RAG_DATA_DIR=/data \
    LLM_PROVIDER=ollama \
    OLLAMA_BASE_URL=http://ollama:11434 \
    OLLAMA_MODEL=llama3.2 \
    OLLAMA_EMBEDDING_MODEL=nomic-embed-text \
    VECTOR_COLLECTION=chat_docs_ollama \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

EXPOSE 7860

CMD ["python", "main.py"]
