FROM python:3.10-slim-bookworm

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    python3-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch FIRST (cached separately from requirements.txt)
# Using torch>=2.6 for CVE-2025-32434 fix
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install torch>=2.6 torchvision>=0.21 --index-url https://download.pytorch.org/whl/cu121

# Now install remaining requirements
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Copy application code
COPY invisible_core/ invisible_core/
COPY app_interface/ app_interface/
COPY scripts/ scripts/

# Create directories
RUN mkdir -p uploads logs models

# Environment variables
ENV PYTHONPATH=/app
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=8080
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501

# Expose both ports
EXPOSE 8080
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app_interface/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
