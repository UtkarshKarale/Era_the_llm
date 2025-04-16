FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY frontend.py .

# Expose ports for FastAPI and Gradio
EXPOSE 8000
EXPOSE 7860

# Create a script to run both services
RUN echo '#!/bin/bash\n\
uvicorn app:app --host 0.0.0.0 --port 8000 & \
python frontend.py\n\
wait' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"] 