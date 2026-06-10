FROM python:3.11-slim

WORKDIR /app

# System dependencies for psycopg2, scikit-learn, Prophet
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy English model locally inside the image
# This means zero internet needed when the container runs
RUN python -m spacy download en_core_web_sm

# Create folders
RUN mkdir -p uploads data/models data/training

# Copy source code
COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]