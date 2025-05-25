FROM python:3.12

WORKDIR /app

# Avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies based on setup guide
RUN apt-get update && apt-get install -y \
    # Basic utilities
    wget \
    curl \
    bzip2 \
    ca-certificates \
    gnupg \
    git \
    # Audio dependencies
    portaudio19-dev \
    libasound2-dev \
    # Graphics and rendering dependencies
    libsdl-pango-dev \
    libcairo2-dev \
    libpango1.0-dev \
    # Media processing
    sox \
    ffmpeg \
    # LaTeX dependencies - TinyTeX (lightweight)
    # Python development tools
    python3-dev \
    build-essential \
    # Additional dependencies
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install TinyTeX
RUN wget -qO- "https://yihui.org/tinytex/install-bin-unix.sh" | sh \
    && ~/.TinyTeX/bin/*/tlmgr path add

# Install common LaTeX packages needed for Manim (use full path to tlmgr)
RUN ~/.TinyTeX/bin/*/tlmgr install amsmath babel-english cbfonts-fd cm-super count1to ctex doublestroke dvisvgm everysel \
    fontspec frcursive fundus-calligra gnu-freefont jknapltx latex-bin mathastext microtype multitoc physics preview prelim2e ragged2e relsize rsfs setspace standalone tipa wasy wasysym xcolor xetex xkeyval

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

# Create conda environment with Python 3.12.8 (as specified in setup guide)
RUN export PATH=/opt/conda/bin:$PATH && conda create -y --name tea python=3.12.8 \
    && conda clean -ay

# Ensure conda is in PATH for all subsequent RUN commands
ENV PATH=/opt/conda/bin:$PATH
SHELL ["/bin/bash", "--login", "-c"]

# Set up conda environment activation
ENV PATH=/opt/conda/envs/tea/bin:$PATH

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies in the conda environment
RUN export PATH=/opt/conda/bin:$PATH && conda run -n tea pip install --no-cache-dir -r requirements.txt

# Create models directory and download TTS models
RUN mkdir -p models \
    && wget -P models https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx \
    && wget -P models https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin

# Copy .env.example for reference
COPY .env.example .

# Copy the application code
COPY . .

# Set PYTHONPATH as specified in setup guide
ENV PYTHONPATH=/app

# Set default Kokoro TTS settings
ENV KOKORO_MODEL_PATH="models/kokoro-v0_19.onnx"
ENV KOKORO_VOICES_PATH="models/voices.bin"
ENV KOKORO_DEFAULT_VOICE="af"
ENV KOKORO_DEFAULT_SPEED="1.0"
ENV KOKORO_DEFAULT_LANG="en-us"

# Create output directory
RUN mkdir -p output

# Expose port for Gradio
EXPOSE 7860

# Health check to verify the container is working
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD conda run -n tea python -c "import src; import manim; print('Health check passed')" || exit 1

# Set the entrypoint command
CMD ["conda", "run", "--no-capture-output", "-n", "tea", "python", "gradio_app.py"]