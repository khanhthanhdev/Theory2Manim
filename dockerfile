FROM ubuntu:22.04

WORKDIR /app

# Avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    ca-certificates \
    git \
    portaudio19-dev \
    libsdl-pango-dev \
    sox \
    ffmpeg \
    texlive-full \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

# Add conda to PATH
ENV PATH /opt/conda/bin:$PATH

# Create conda environment
RUN conda create -y --name tea python=3.12 \
    && echo "source activate tea" > ~/.bashrc

# Set up conda environment activation
ENV PATH /opt/conda/envs/tea/bin:$PATH
SHELL ["/bin/bash", "--login", "-c"]

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN conda run -n tea pip install -r requirements.txt

# Copy the application code
COPY . .

# Set PYTHONPATH
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose port for Gradio
EXPOSE 7860

# Set the entrypoint command
CMD ["conda", "run", "--no-capture-output", "-n", "tea", "python", "gradio_app.py"]