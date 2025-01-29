# Use an official Python base image
FROM python:3.10-slim

# Create and set the working directory
WORKDIR /app

# Install system packages and pip explicitly
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies directly
# Alternatively, you can COPY a requirements.txt and do `pip install -r requirements.txt`
RUN pip install --no-cache-dir \
    streamlit \
    mlflow \
    python-dotenv \
    faiss-cpu \
    langchain \
    langchain-groq \
    huggingface_hub

# Copy your application code and resource files into the container
COPY . /app

# Expose Streamlit's default port
EXPOSE 8501

# By default, run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
