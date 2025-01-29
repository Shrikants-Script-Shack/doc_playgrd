FROM python:3.10-slim


WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --no-cache-dir \
    streamlit \
    python-dotenv \
    faiss-cpu \
    langchain \
    langchain-groq \
    huggingface_hub \
	langchain-community

COPY . /app

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
