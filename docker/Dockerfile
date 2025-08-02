# Dockerfile for Differentially Private Federated Learning Simulation
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY run_pipeline.py .
COPY app/ ./app/

# Create necessary directories
RUN mkdir -p data/raw data/processed data/external \
    models visualizations logs tests notebooks

# Copy notebooks
COPY notebooks/ ./notebooks/

# Copy tests
COPY tests/ ./tests/

# Copy configuration files
COPY .gitignore .
COPY README.md .
COPY LICENSE .

# Set permissions
RUN chmod +x run_pipeline.py

# Expose port for Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"] 