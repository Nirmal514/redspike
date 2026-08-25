FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for audio / DSP
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train and pre-export model weights on build
RUN python ml_engine/train.py

EXPOSE 8000

CMD ["uvicorn", "dashboard.app:app", "--host", "0.0.0.0", "--port", "8000"]
