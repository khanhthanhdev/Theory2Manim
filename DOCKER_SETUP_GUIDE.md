# Docker Setup Guide for TheoremExplainAgent 🐳

This guide provides instructions for running TheoremExplainAgent using Docker, which simplifies the setup process and ensures consistency across different environments.

## Prerequisites

- **Docker** (version 20.10 or higher)
- **Docker Compose** (version 2.0 or higher)
- At least 8GB of available disk space
- At least 4GB of RAM

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/TIGER-AI-Lab/TheoremExplainAgent.git
cd TheoremExplainAgent
```

### 2. Configure Environment Variables
```bash
# Copy the environment template
cp .env.template .env

# Edit the .env file with your API keys
nano .env  # or use your preferred editor
```

Add your API keys to the `.env` file:
```bash
# At minimum, add one of these API keys:
OPENAI_API_KEY="your_openai_api_key_here"
# OR
GEMINI_API_KEY="your_gemini_api_key_here"

# Kokoro TTS settings (pre-configured)
KOKORO_MODEL_PATH="models/kokoro-v0_19.onnx"
KOKORO_VOICES_PATH="models/voices.bin"
KOKORO_DEFAULT_VOICE="af"
KOKORO_DEFAULT_SPEED="1.0"
KOKORO_DEFAULT_LANG="en-us"
```

### 3. Build and Run with Docker Compose
```bash
# Build and start the container
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

The Gradio web interface will be available at: http://localhost:7860

### 4. Alternative: Manual Docker Commands
```bash
# Build the Docker image
docker build -t theoremexplain-agent .

# Run the container
docker run -d \
  --name theoremexplain \
  -p 7860:7860 \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  --env-file .env \
  theoremexplain-agent
```

## Docker Features

### Included in the Docker Image:
- ✅ Ubuntu 22.04 base with all system dependencies
- ✅ Python 3.12.8 in a Conda environment
- ✅ All Python packages from requirements.txt
- ✅ MiKTeX (lightweight LaTeX distribution) for mathematical rendering
- ✅ FFmpeg for video processing
- ✅ Audio libraries (portaudio, sox)
- ✅ Kokoro TTS models automatically downloaded
- ✅ Health checks for container monitoring

### LaTeX Setup (MiKTeX):
The Docker image uses MiKTeX instead of texlive-full to reduce image size:
- **Size**: ~500MB vs ~3GB (texlive-full)
- **Auto-install**: Missing packages are installed automatically
- **Pre-installed**: Common math packages (amsmath, amsfonts, amssymb, geometry, xcolor, graphicx)
- **Flexibility**: Easy to add more packages as needed

### Volume Mounts:
- `./output:/app/output` - Persists generated videos
- `./models:/app/models` - TTS models storage
- `./data:/app/data` - Datasets and RAG documents

## Usage Examples

### Using the Web Interface
1. Access http://localhost:7860 in your browser
2. Enter a topic (e.g., "Pythagorean Theorem")
3. Add context description
4. Select AI model
5. Generate video

### Command Line Generation
```bash
# Generate a single video
docker exec -it theoremexplain conda run -n tea python generate_video.py \
  --model "openai/gpt-4o-mini" \
  --helper_model "openai/gpt-4o-mini" \
  --output_dir "output/docker_test" \
  --topic "Quadratic Formula" \
  --context "Formula for solving quadratic equations"

# Batch generation
docker exec -it theoremexplain conda run -n tea python generate_video.py \
  --model "openai/gpt-4o-mini" \
  --helper_model "openai/gpt-4o-mini" \
  --output_dir "output/batch_docker" \
  --theorems_path "data/thb_easy/math.json" \
  --sample_size 3
```

### Batch Generation Service
Run batch generation as a separate service:
```bash
# Start batch generation service
docker-compose --profile batch up batch-theoremexplain

# Or customize the batch command
docker-compose run --rm theoremexplain-batch conda run -n tea python generate_video.py \
  --model "gemini/gemini-1.5-flash" \
  --helper_model "gemini/gemini-1.5-flash" \
  --output_dir "output/gemini_batch" \
  --topic "Derivatives" \
  --context "Calculus concept for rate of change"
```

## Monitoring and Logs

### View Container Logs
```bash
# View live logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f theoremexplain
```

### Container Health Check
```bash
# Check container status
docker-compose ps

# Manual health check
docker exec theoremexplain conda run -n tea python -c "import src; import manim; print('Container is healthy')"
```

### Resource Monitoring
```bash
# Monitor resource usage
docker stats theoremexplain-agent

# View detailed container info
docker inspect theoremexplain-agent
```

## Troubleshooting

### Common Issues:

#### 1. Container fails to start
```bash
# Check build logs
docker-compose build --no-cache

# Check container logs
docker-compose logs
```

#### 2. Permission issues with volumes
```bash
# Fix permissions on Linux/macOS
sudo chown -R $USER:$USER output/ models/ data/

# On Windows with WSL
wsl sudo chown -R $(whoami):$(whoami) output/ models/ data/
```

#### 3. Out of memory errors
```bash
# Increase Docker memory limit to 8GB or more
# Edit Docker Desktop settings or daemon.json
```

#### 4. TTS models not downloading
```bash
# Manually download models
docker exec -it theoremexplain bash
wget -P models https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx
wget -P models https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin
```

#### 5. API key issues
```bash
# Verify environment variables
docker exec theoremexplain env | grep API

# Test API connection
docker exec theoremexplain conda run -n tea python -c "
import os
from openai import OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
print('API key is working!')
"
```

#### 6. LaTeX/MiKTeX issues
```bash
# Check MiKTeX installation
docker exec theoremexplain miktex --version

# Install additional LaTeX packages if needed
docker exec theoremexplain mpm --admin --install <package-name>

# Update MiKTeX package database
docker exec theoremexplain mpm --admin --update-db
```

## Docker Management

### Update the Application
```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose up --build -d
```

### Clean Up
```bash
# Stop and remove containers
docker-compose down

# Remove images
docker rmi theoremexplain-agent

# Clean up unused Docker resources
docker system prune -f
```

### Backup Generated Videos
```bash
# Create backup of output directory
tar -czf theoremexplain-backup-$(date +%Y%m%d).tar.gz output/

# Or copy to external location
cp -r output/ /path/to/backup/location/
```

## Performance Optimization

### For Better Performance:
1. **Increase Docker Resources**: Allocate 8GB+ RAM and 4+ CPU cores
2. **Use SSD Storage**: Mount output directory on SSD for faster video rendering
3. **GPU Support**: Add GPU support for faster rendering (requires nvidia-docker)

### GPU Support (Optional):
```yaml
# Add to docker-compose.yml
services:
  theoremexplain:
    # ... existing config ...
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Advanced Configuration

### Custom Models Directory:
```bash
# Use external models directory
docker run -v /path/to/models:/app/models theoremexplain-agent
```

### RAG Setup with Docker:
```bash
# Download RAG docs to data/rag/manim_docs/
# Then run with RAG enabled
docker exec theoremexplain conda run -n tea python generate_video.py \
  --model "openai/gpt-4o-mini" \
  --topic "Animations" \
  --context "Manim animation concepts" \
  --use_rag \
  --chroma_db_path "data/rag/chroma_db" \
  --manim_docs_path "data/rag/manim_docs"
```

## Security Notes

- Never commit `.env` files with real API keys to version control
- Use Docker secrets for production deployments
- Regularly update the base image for security patches
- Consider using multi-stage builds for smaller production images

## Support

If you encounter issues with Docker setup:
1. Check the main [SETUP_GUIDE.md](./SETUP_GUIDE.md) for general troubleshooting
2. Verify your Docker installation: `docker --version` and `docker-compose --version`
3. Check Docker daemon status and available resources
4. Create an issue on the GitHub repository with Docker logs

---

**Happy containerized theorem explaining! 🐳✨**
