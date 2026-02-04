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

# Copy font file
COPY OpenSans-Bold.ttf /app/OpenSans-Bold.ttf

# Install PyTorch FIRST (cached separately from requirements.txt)
# Using torch>=2.6 for CVE-2025-32434 fix
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch>=2.6 torchvision>=0.21 --index-url https://download.pytorch.org/whl/cu121

# Now install remaining requirements (changes here won't re-download PyTorch)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# PHASE 5.2 DELTA: Install new libs (Cached separately to avoid re-downloading heavy requirements.txt)
COPY requirements_delta.txt .
RUN pip install --no-cache-dir -r requirements_delta.txt

COPY src/ src/

# Create uploads directory
RUN mkdir uploads

# Copy verification scripts
COPY verify_protocols.py .
COPY validator.py .
COPY validator_stealth.py .
COPY verify_stealth_pipeline.py .
COPY verify_comparison.py .
COPY verify_project_invisible.py .
COPY reproduce_issue.py .

# Environment variables
ENV PYTHONPATH=/app
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=8080
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501

# Expose both ports (Gradio 8080, Streamlit 8501)
EXPOSE 8080
EXPOSE 8501

# Run the Streamlit app by default (use `python src/main.py` for Gradio)
CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
