# TheoremExplainAgent (TEA) Setup Guide 🍵

This guide will walk you through setting up the TheoremExplainAgent project from scratch, including cloning the repository, installing dependencies, and running your first video generation.

## Prerequisites

Before starting, ensure you have the following installed on your system:

- **Python 3.12.8** (recommended)
- **Git**
- **Conda** or **Miniconda**
- **LaTeX** distribution (for mathematical rendering)
- **FFmpeg** (for video processing)

### System-Specific Prerequisites

#### Windows (WSL recommended)
```bash
# Install WSL if not already installed
wsl --install

# In WSL, install required packages
sudo apt-get update
sudo apt-get install portaudio19-dev
sudo apt-get install libsdl-pango-dev
sudo apt-get install texlive-full  # LaTeX distribution
sudo apt-get install ffmpeg
```

#### macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required packages
brew install portaudio
brew install ffmpeg
brew install --cask mactex  # LaTeX distribution
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install portaudio19-dev
sudo apt-get install libsdl-pango-dev
sudo apt-get install texlive-full
sudo apt-get install ffmpeg
sudo apt-get install python3-dev
sudo apt-get install build-essential
```

## Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/TIGER-AI-Lab/TheoremExplainAgent.git

# Navigate to the project directory
cd TheoremExplainAgent
```

## Step 2: Set Up Python Environment

### Create Conda Environment
```bash
# Create a new conda environment with Python 3.12.8
conda create --name tea python=3.12.8

# Activate the environment
conda activate tea
```

### Alternative: Using Python venv
```bash
# Create virtual environment
python -m venv tea_env

# Activate the environment
# On Windows (WSL/Git Bash):
source tea_env/bin/activate
# On Windows (Command Prompt):
tea_env\Scripts\activate
# On macOS/Linux:
source tea_env/bin/activate
```

## Step 3: Install Python Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### If you encounter installation issues:

#### For audio processing issues:
```bash
# On Ubuntu/Debian
sudo apt-get install libasound2-dev

# On macOS
brew install portaudio

# Then retry pip install
pip install pyaudio
```

#### For Cairo/Pango issues:
```bash
# On Ubuntu/Debian
sudo apt-get install libcairo2-dev libpango1.0-dev

# On macOS
brew install cairo pango

# Then retry pip install
pip install pycairo
```

## Step 4: Download TTS Models

Download the Kokoro TTS models for voice generation:

```bash
# Create models directory and download files
mkdir -p models
wget -P models https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx
wget -P models https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin
```

### Alternative download (if wget is not available):
1. Go to [Kokoro ONNX Releases](https://github.com/thewh1teagle/kokoro-onnx/releases)
2. Download `kokoro-v0_19.onnx` and `voices.bin`
3. Place them in the `models/` directory

## Step 5: Configure Environment Variables

### Create .env file
```bash
# Copy the template
cp .env.template .env

# Edit the .env file with your preferred text editor
nano .env  # or vim .env, or code .env
```

### Configure API Keys
Add your API keys to the `.env` file. Here's a template:

```bash
# OpenAI (recommended for beginners)
OPENAI_API_KEY="your_openai_api_key_here"

# Azure OpenAI (if using Azure)
AZURE_API_KEY=""
AZURE_API_BASE=""
AZURE_API_VERSION=""

# Google Vertex AI (if using Google Cloud)
VERTEXAI_PROJECT=""
VERTEXAI_LOCATION=""
GOOGLE_APPLICATION_CREDENTIALS=""

# Google Gemini (if using Gemini directly)
GEMINI_API_KEY=""

# Kokoro TTS Settings (pre-configured)
KOKORO_MODEL_PATH="models/kokoro-v0_19.onnx"
KOKORO_VOICES_PATH="models/voices.bin"
KOKORO_DEFAULT_VOICE="af"
KOKORO_DEFAULT_SPEED="1.0"
KOKORO_DEFAULT_LANG="en-us"
```

### Get API Keys:
- **OpenAI**: Visit [OpenAI API](https://platform.openai.com/api-keys)
- **Google Gemini**: Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
- **Azure OpenAI**: Set up through [Azure Portal](https://portal.azure.com/)

## Step 6: Configure Python Path

This is crucial for the project to work correctly:

```bash
# Add current directory to Python path
export PYTHONPATH=$(pwd):$PYTHONPATH

# To make this permanent, add to your shell profile
echo 'export PYTHONPATH=$(pwd):$PYTHONPATH' >> ~/.bashrc  # for bash
echo 'export PYTHONPATH=$(pwd):$PYTHONPATH' >> ~/.zshrc   # for zsh
```

## Step 7: Verify Installation

### Test basic installation:
```bash
# Test Python imports
python -c "import manim; import src; print('Installation successful!')"
```

### Test TTS models:
```bash
# Check if TTS models are accessible
python -c "
import os
from src.utils.kokoro_voiceover import KokoroService
print('TTS models found and accessible!')
"
```

## Step 8: Run Your First Video Generation

### Simple single topic generation:
```bash
python generate_video.py \
    --model "openai/gpt-4o-mini" \
    --helper_model "openai/gpt-4o-mini" \
    --output_dir "output/my_first_video" \
    --topic "Pythagorean Theorem" \
    --context "A fundamental theorem in geometry about right triangles"
```

### Using Gemini (if you have Gemini API key):
```bash
python generate_video.py \
    --model "gemini/gemini-1.5-flash" \
    --helper_model "gemini/gemini-1.5-flash" \
    --output_dir "output/gemini_test" \
    --topic "Quadratic Formula" \
    --context "Formula for finding roots of quadratic equations"
```

## Optional: Set Up RAG (Retrieval Augmented Generation)

For enhanced generation with documentation context:

### Download RAG documentation:
1. Download from [Google Drive](https://drive.google.com/file/d/1Tn6J_JKVefFZRgZbjns93KLBtI9ullRv/view?usp=sharing)
2. Extract to `data/rag/manim_docs/`

### Generate with RAG:
```bash
python generate_video.py \
    --model "openai/gpt-4o-mini" \
    --helper_model "openai/gpt-4o-mini" \
    --output_dir "output/with_rag" \
    --topic "Big O Notation" \
    --context "Asymptotic notation for algorithm complexity" \
    --use_rag \
    --chroma_db_path "data/rag/chroma_db" \
    --manim_docs_path "data/rag/manim_docs" \
    --embedding_model "vertex_ai/text-embedding-005"
```

## Troubleshooting

### Common Issues and Solutions:

#### 1. ModuleNotFoundError: No module named 'src'
```bash
# Solution: Set Python path
export PYTHONPATH=$(pwd):$PYTHONPATH
```

#### 2. LaTeX not found errors
```bash
# Install LaTeX distribution
# Ubuntu/Debian:
sudo apt-get install texlive-full

# macOS:
brew install --cask mactex

# Windows: Install MiKTeX or TeX Live
```

#### 3. Audio/TTS issues
```bash
# Install audio dependencies
# Ubuntu/Debian:
sudo apt-get install portaudio19-dev libasound2-dev

# macOS:
brew install portaudio
```

#### 4. FFmpeg not found
```bash
# Install FFmpeg
# Ubuntu/Debian:
sudo apt-get install ffmpeg

# macOS:
brew install ffmpeg

# Windows: Download from https://ffmpeg.org/
```

#### 5. API rate limits or errors
- Check your API keys in `.env`
- Verify API quotas and billing
- Try using a different model (e.g., switch from gpt-4 to gpt-3.5-turbo)

### Getting Help:

1. **Check the FAQ** in the main README.md
2. **Enable debug mode** by setting environment variable:
   ```bash
   export LITELLM_LOG=DEBUG
   ```
3. **Create an issue** on the GitHub repository if problems persist

## Next Steps

Once setup is complete:

1. **Explore different topics**: Try generating videos for various mathematical concepts
2. **Experiment with models**: Test different AI models to see which works best for your use case
3. **Try batch generation**: Use the `--theorems_path` option to generate multiple videos
4. **Customize prompts**: Modify prompts in `task_generator/prompts_raw/` for different styles

## Project Structure Overview

```
TheoremExplainAgent/
├── src/                    # Core source code
├── task_generator/         # Prompt templates and generation logic
├── eval_suite/            # Evaluation tools
├── data/                  # Datasets and RAG documents
├── models/                # TTS models
├── output/                # Generated videos
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (you create this)
└── generate_video.py      # Main generation script
```

Happy theorem explaining! 🎓✨
