# Use an official Python 3.10 slim image
FROM python:3.10-slim

# Set a working directory
WORKDIR /app

# Install pip and any needed system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
# If you prefer, you can COPY a requirements.txt and do: pip install -r requirements.txt
RUN pip install --no-cache-dir \
    streamlit \
    python-dotenv \
    faiss-cpu \
    langchain \
    langchain-groq \
    huggingface_hub \
	langchain-community

# Copy all files into the container
COPY . /app

# Expose Streamlit's default port
EXPOSE 8501

# Final command: run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
