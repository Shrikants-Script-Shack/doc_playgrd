# Use the official Python image
FROM python:3.11

# Set the working directory inside the container
WORKDIR /app

# Copy requirements (optional, if using requirements.txt)
COPY requirements.txt .

# Install FastAPI and Uvicorn
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the container
COPY . .

# Expose port 8000 for FastAPI
EXPOSE 8000

# Command to run the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
