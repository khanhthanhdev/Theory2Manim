# TheoremExplainAgent Script Usage Guide

## Table of Contents
1. [Quick Start](#quick-start)
2. [Running Without RAG](#running-without-rag)
3. [Running With RAG](#running-with-rag)
4. [Performance Optimization](#performance-optimization)
5. [Command-Line Arguments Reference](#command-line-arguments-reference)
6. [Examples](#examples)
7. [Troubleshooting](#troubleshooting)

## Quick Start

The basic syntax for running the script is:
```bash
python generate_video.py [OPTIONS]
```

Ensure you have:
1. Set up your environment according to the [README.md](README.md)
2. Configured your `.env` file with API keys
3. Activated the conda environment: `conda activate tea`
4. Set Python path: `export PYTHONPATH=$(pwd):$PYTHONPATH`

## Running Without RAG

### Single Topic Generation
Generate a video for a single topic without RAG (Retrieval Augmented Generation):

```bash
python generate_video.py \
    --model "openai/o3-mini" \
    --helper_model "openai/o3-mini" \
    --output_dir "output/my_experiment" \
    --topic "Big O notation" \
    --context "most common type of asymptotic notation in computer science used to measure worst case complexity"
```

### Batch Processing
Process multiple topics from a JSON file:

```bash
python generate_video.py \
    --model "openai/o3-mini" \
    --helper_model "openai/o3-mini" \
    --output_dir "output/batch_processing" \
    --theorems_path "data/thb_easy/math.json" \
    --sample_size 10 \
    --max_scene_concurrency 5 \
    --max_topic_concurrency 2
```

## Running With RAG

RAG (Retrieval Augmented Generation) enhances the system by providing relevant Manim documentation context during code generation.

### Prerequisites for RAG
1. Download RAG documentation from [Google Drive](https://drive.google.com/file/d/1Tn6J_JKVefFZRgZbjns93KLBtI9ullRv/view?usp=sharing)
2. Extract to a directory (e.g., `data/rag/manim_docs`)
3. Vector database will be created automatically on first run

### Single Topic with RAG
```bash
python generate_video.py \
    --model "openai/o3-mini" \
    --helper_model "openai/o3-mini" \
    --output_dir "output/with_rag/single_topic" \
    --topic "Pythagorean theorem" \
    --context "fundamental theorem in geometry relating to right triangles" \
    --use_rag \
    --chroma_db_path "data/rag/chroma_db" \
    --manim_docs_path "data/rag/manim_docs" \
    --embedding_model "vertex_ai/text-embedding-005"
```

### Batch Processing with RAG
```bash
python generate_video.py \
    --model "openai/o3-mini" \
    --helper_model "openai/o3-mini" \
    --output_dir "output/with_rag/batch" \
    --theorems_path "data/thb_easy/math.json" \
    --use_rag \
    --chroma_db_path "data/rag/chroma_db" \
    --manim_docs_path "data/rag/manim_docs" \
    --embedding_model "vertex_ai/text-embedding-005" \
    --max_scene_concurrency 7 \
    --max_topic_concurrency 20
```

## Performance Optimization

### GPU Acceleration
Enable GPU acceleration for faster video processing:

```bash
python generate_video.py \
    --model "openai/o3-mini" \
    --topic "Linear transformations" \
    --use_gpu_acceleration \
    --quality "high" \
    --max_concurrent_renders 8
```

**Note**: GPU acceleration requires:
- NVIDIA GPU with NVENC support
- Proper FFmpeg installation with NVIDIA codec support
- Sufficient VRAM

### Caching for Faster Development
Enable intelligent caching to avoid re-rendering identical scenes:

```bash
python generate_video.py \
    --model "openai/o3-mini" \
    --topic "Fibonacci sequence" \
    --enable_caching \
    --quality "medium"
```

### Preview Mode for Development
Use preview mode for faster iteration during development:

```bash
python generate_video.py \
    --model "openai/o3-mini" \
    --topic "Prime numbers" \
    --preview_mode \
    --quality "preview"
```

### Concurrency Settings
Optimize concurrent processing based on your hardware:

```bash
# For high-end systems
python generate_video.py \
    --model "openai/o3-mini" \
    --theorems_path "data/thb_easy/math.json" \
    --max_scene_concurrency 10 \
    --max_topic_concurrency 5 \
    --max_concurrent_renders 8

# For modest systems
python generate_video.py \
    --model "openai/o3-mini" \
    --theorems_path "data/thb_easy/math.json" \
    --max_scene_concurrency 3 \
    --max_topic_concurrency 1 \
    --max_concurrent_renders 2
```

## Command-Line Arguments Reference

### Core Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | `gemini/gemini-2.5-flash-preview-04-17` | Primary AI model for generation |
| `--scene_model` | str | None | Specific model for scene generation |
| `--helper_model` | str | None | Helper model for additional tasks |

### Input/Output

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--topic` | str | - | Single topic to process |
| `--context` | str | - | Context description for the topic |
| `--theorems_path` | str | - | Path to theorems JSON file for batch processing |
| `--output_dir` | str | `output` | Output directory for generated content |

### Processing Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--sample_size` | int | - | Number of theorems to sample from JSON |
| `--scenes` | list[int] | - | Specific scene numbers to process |
| `--max_retries` | int | 5 | Maximum retries for code generation |

### Mode Flags

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--only_plan` | flag | False | Only generate plans, skip rendering |
| `--only_render` | flag | False | Only render existing scenes |
| `--only_combine` | flag | False | Only combine existing videos |
| `--check_status` | flag | False | Check status of all topics |

### Performance Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--max_scene_concurrency` | int | 5 | Maximum concurrent scene processing |
| `--max_topic_concurrency` | int | 1 | Maximum concurrent topic processing |
| `--max_concurrent_renders` | int | 4 | Maximum concurrent render processes |
| `--quality` | str | `medium` | Render quality: `preview`, `low`, `medium`, `high`, `production` |

### Feature Flags

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--verbose` | flag | False | Enable verbose output |
| `--use_rag` | flag | False | Enable RAG (Retrieval Augmented Generation) |
| `--use_context_learning` | flag | False | Enable context learning |
| `--use_visual_fix_code` | flag | False | Enable visual code fixing |
| `--use_langfuse` | flag | False | Enable Langfuse logging |
| `--enable_caching` | flag | True | Enable intelligent caching |
| `--use_gpu_acceleration` | flag | False | Enable GPU acceleration |
| `--preview_mode` | flag | False | Enable preview mode |

### RAG Configuration

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--chroma_db_path` | str | `data/rag/chroma_db` | Path to ChromaDB database |
| `--manim_docs_path` | str | `data/rag/manim_docs` | Path to Manim documentation |
| `--embedding_model` | str | `hf:ibm-granite/granite-embedding-30m-english` | Embedding model for RAG |
| `--context_learning_path` | str | `data/context_learning` | Path to context learning examples |

## Examples

### Example 1: Basic Single Topic
```bash
python generate_video.py \
    --model "openai/o3-mini" \
    --topic "Derivative rules" \
    --context "basic rules for computing derivatives in calculus"
```

### Example 2: High-Quality Production with GPU
```bash
python generate_video.py \
    --model "openai/o3-mini" \
    --topic "Matrix multiplication" \
    --context "mathematical operation for combining matrices" \
    --quality "production" \
    --use_gpu_acceleration \
    --max_concurrent_renders 6
```

### Example 3: Development Mode with RAG
```bash
python generate_video.py \
    --model "openai/o3-mini" \
    --topic "Graph algorithms" \
    --context "algorithms for traversing and analyzing graphs" \
    --use_rag \
    --preview_mode \
    --enable_caching \
    --verbose
```

### Example 4: Batch Processing with All Features
```bash
python generate_video.py \
    --model "openai/o3-mini" \
    --helper_model "openai/o3-mini" \
    --theorems_path "data/thb_easy/math.json" \
    --output_dir "output/full_batch" \
    --use_rag \
    --use_visual_fix_code \
    --enable_caching \
    --use_gpu_acceleration \
    --quality "high" \
    --max_scene_concurrency 8 \
    --max_topic_concurrency 3 \
    --verbose
```

### Example 5: Status Check
```bash
python generate_video.py \
    --theorems_path "data/thb_easy/math.json" \
    --check_status
```

### Example 6: Specific Scenes Only
```bash
python generate_video.py \
    --model "openai/o3-mini" \
    --topic "Quadratic formula" \
    --scenes 1 3 5 \
    --quality "medium"
```

## Quality Presets

| Quality | Resolution | FPS | Use Case |
|---------|-----------|-----|----------|
| `preview` | 480p | 15 | Quick development/testing |
| `low` | 480p | 15 | Fast rendering |
| `medium` | 720p | 30 | Balanced quality/speed |
| `high` | 1080p | 60 | High quality output |
| `production` | 1440p | 60 | Maximum quality |

## Performance Guidelines

### System Requirements by Configuration

#### Basic Usage (No GPU)
- CPU: 4+ cores
- RAM: 8GB+
- Concurrency: `--max_scene_concurrency 3 --max_topic_concurrency 1`

#### Moderate Performance
- CPU: 8+ cores
- RAM: 16GB+
- Concurrency: `--max_scene_concurrency 5 --max_topic_concurrency 2`

#### High Performance (with GPU)
- CPU: 12+ cores
- RAM: 32GB+
- GPU: NVIDIA RTX series with 8GB+ VRAM
- Concurrency: `--max_scene_concurrency 10 --max_topic_concurrency 5`

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   export PYTHONPATH=$(pwd):$PYTHONPATH
   ```

2. **API Key Issues**
   - Check your `.env` file configuration
   - Verify API keys are valid
   - Enable `--verbose` for debugging

3. **Memory Issues**
   - Reduce concurrency settings
   - Use lower quality presets
   - Enable `--preview_mode`

4. **GPU Acceleration Not Working**
   - Verify NVIDIA drivers and CUDA installation
   - Check FFmpeg NVENC support: `ffmpeg -encoders | grep nvenc`
   - Fallback to CPU: remove `--use_gpu_acceleration`

5. **RAG Setup Issues**
   - Ensure manim docs are downloaded and extracted
   - Check ChromaDB path permissions
   - Verify embedding model availability

### Debug Mode
Enable maximum verbosity for troubleshooting:
```bash
python generate_video.py \
    --model "openai/o3-mini" \
    --topic "Test topic" \
    --verbose \
    --preview_mode \
    --max_scene_concurrency 1
```

## Advanced Usage

### Custom Model Configuration
```bash
python generate_video.py \
    --model "anthropic/claude-3-5-sonnet-20241022" \
    --scene_model "openai/o3-mini" \
    --helper_model "gemini/gemini-1.5-pro-002" \
    --topic "Advanced calculus"
```

### Workflow Combinations
```bash
# Step 1: Generate plans only
python generate_video.py \
    --model "openai/o3-mini" \
    --theorems_path "data/thb_easy/math.json" \
    --only_plan

# Step 2: Render specific scenes
python generate_video.py \
    --theorems_path "data/thb_easy/math.json" \
    --only_render \
    --scenes 1 2 3

# Step 3: Combine videos
python generate_video.py \
    --theorems_path "data/thb_easy/math.json" \
    --only_combine
```

This comprehensive guide should help you effectively use the TheoremExplainAgent script for generating educational mathematics videos with various configurations and optimizations.
