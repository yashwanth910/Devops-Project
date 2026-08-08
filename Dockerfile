FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and the registered/exported model
COPY src/ ./src/
COPY models/ ./models/

WORKDIR /app/src

EXPOSE 8000

# Startup configuration
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
