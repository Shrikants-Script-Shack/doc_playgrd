# Use official Python image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all necessary files
COPY . .

# Ensure .env is available inside the container
COPY .env .env

# Expose required ports
EXPOSE 8501 5000  # Streamlit & MLflow

# Start MLflow first, wait, then run Streamlit
CMD ["sh", "-c", "mlflow server --host 0.0.0.0 --port 5000 & sleep 5 && streamlit run dockerapp_key.py --server.port=8501 --server.address=0.0.0.0"]
