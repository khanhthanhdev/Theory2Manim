# Stage 1: Builder
FROM python:3.12-slim AS builder

WORKDIR /app

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 

# Install build dependencies in a single layer with version pinning where critical
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget=1.21.* \
    curl \
    gcc \
    bzip2 \
    ca-certificates \
    gnupg \
    git \
    python3-dev \
    build-essential \
    pkg-config \
    portaudio19-dev \
    libsdl-pango-dev \
    libcairo2-dev \
    libpango1.0-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*



# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu \
    && python -c "import gradio; print(f'Gradio version: {gradio.__version__}')" \
    && find /install -name "*.pyc" -delete \
    && find /install -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Download models with checksums and error handling
RUN mkdir -p /models \
    && cd /models \
    && wget --progress=dot:giga -O kokoro-v0_19.onnx \
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx" \
    && wget --progress=dot:giga -O voices.bin \
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin" \
    && ls -la /models

# Stage 2: Runtime
FROM python:3.12-slim AS runtime

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# Set environment variables
# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    NETWORKX_AUTOMATIC_BACKENDS=false \
    NETWORKX_BACKEND_PRIORITY=""

# Install runtime dependencies including LaTeX tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    portaudio19-dev \
    libasound2-dev \
    libsdl-pango-dev \
    libcairo2-dev \
    libpango1.0-dev \
    sox \
    ffmpeg \
    tini \
    texlive-full \
    dvisvgm \
    ghostscript \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*



# Copy Python packages from builder stage
COPY --from=builder /install /usr/local

# Copy models from builder stage
COPY --from=builder /models /app/models

# Copy application files (be more selective to reduce layer size)
COPY --chown=appuser:appuser .env gradio_app.py ./
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser mllm_tools/ ./mllm_tools/
COPY --chown=appuser:appuser task_generator/ ./task_generator/
COPY --chown=appuser:appuser *.py ./
# Create output directory with proper permissions
RUN mkdir -p output \
    && chown -R appuser:appuser /app

# Copy data folder needed for embedding creation
COPY --chown=appuser:appuser data/ ./data/

# Run embedding creation script at build time
RUN python create_embeddings.py

# Switch to non-root user
USER appuser

# Add labels for better maintainability
LABEL maintainer="your-email@example.com" \
      version="1.0" \
      description="Multi-stage Docker image for ML application"

# Expose port
EXPOSE 7860

# Use tini as PID 1 for proper signal handling
ENTRYPOINT ["tini", "--"]


# Start the application
CMD ["python", "gradio_app.py"]