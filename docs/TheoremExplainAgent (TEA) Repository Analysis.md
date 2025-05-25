# TheoremExplainAgent (TEA) Repository Analysis

## Overview
TheoremExplainAgent is an AI system that generates long-form Manim videos to visually explain theorems, demonstrating deep understanding while revealing reasoning flaws that text alone might hide. The project was accepted to ACL 2025 main conference and uses a multi-agent approach to create educational science videos from mathematical, physical, and chemical topics.

## Repository Structure

The repository is organized as follows:

- **src/**: Core source code for the project
- **data/**: Contains theorem datasets and other data resources
- **eval_suite/**: Evaluation tools and metrics
- **mllm_tools/**: Multimodal language model tools
- **task_generator/**: Tools for generating tasks/topics
- **generate_video.py**: Main script for generating explanation videos
- **evaluate.py**: Script for evaluating the generated content
- **requirements.txt**: Dependencies for the project
- **.env.template**: Template for environment variables configuration

## Key Components Analysis

### 1. Backend Architecture

The backend of TheoremExplainAgent consists of:

- **LLM Integration**: Uses LiteLLM for model management, supporting various models:
  - OpenAI models
  - Azure OpenAI
  - Google Vertex AI
  - Google Gemini
  - Other compatible models

- **Theorem Processing Pipeline**:
  - Takes theorem topics as input
  - Processes them through LLM agents
  - Generates Manim code for visualization
  - Renders videos with audio narration

- **Concurrency Management**:
  - Supports parallel processing with `--max_scene_concurrency` and `--max_topic_concurrency` parameters
  - Enables batch processing of multiple theorems

- **TTS Integration**:
  - Uses Kokoro TTS for voice generation
  - Configurable voice settings (speed, language, voice type)

### 2. Frontend Considerations

While the repository doesn't have an explicit frontend component, the output is visual content:

- **Video Output**: 
  - Generates Manim animations as the primary output
  - Creates multimodal explanations combining visuals and audio

- **Potential Frontend Development Areas**:
  - Web interface for submitting theorem topics
  - Video player with interactive controls
  - Dashboard for tracking generation progress
  - Result visualization and management interface

### 3. Video Compiler

The video compilation process involves:

- **Manim Integration**:
  - Uses Manim Community for mathematical animations
  - Requires LaTeX and other dependencies for rendering
  - Generates scene-by-scene animations

- **Audio-Visual Synchronization**:
  - Combines generated animations with TTS audio
  - Creates cohesive educational videos

- **Current Performance Considerations**:
  - Processing is resource-intensive
  - Uses concurrency parameters to manage resource allocation
  - Separate steps for generation and combination

### 4. Model Architecture

The system employs a multi-agent approach:

- **Primary Model**: 
  - Main LLM for theorem understanding and explanation generation
  - Configurable via `--model` parameter (e.g., "openai/o3-mini")

- **Helper Model**:
  - Secondary LLM for supporting tasks
  - Configurable via `--helper_model` parameter

- **Embedding Model**:
  - Used for RAG functionality
  - Options include "azure/text-embedding-3-large" and "vertex_ai/text-embedding-005"

- **TTS Model**:
  - Kokoro model for text-to-speech conversion
  - Pre-trained model files required

### 5. RAG (Retrieval Augmented Generation)

The repository includes RAG capabilities:

- **Documentation Retrieval**:
  - Downloads Manim documentation for reference
  - Creates vector database for efficient retrieval

- **Vector Database**:
  - Uses ChromaDB for storing embeddings
  - Created on first run with RAG enabled

- **RAG Configuration**:
  - Enabled via `--use_rag` flag
  - Requires setting paths for ChromaDB and Manim documentation
  - Configurable embedding model

- **Context Learning**:
  - Optional feature via `--use_context_learning`
  - Stores and retrieves context for improved generation

## Current Limitations and Optimization Opportunities

1. **Video Compiler Speed**:
   - Rendering is computationally intensive
   - Limited concurrency options
   - No explicit caching mechanism for intermediate results

2. **Model Architecture**:
   - Fixed agent roles and interactions
   - Limited model selection documentation
   - No explicit fine-tuning capabilities

3. **RAG Implementation**:
   - Basic retrieval mechanism
   - Limited to Manim documentation
   - No apparent domain-specific optimization

4. **Frontend**:
   - No user interface for interaction
   - Command-line only operation
   - Limited visualization of generation progress

5. **Backend**:
   - Limited error handling and recovery
   - No distributed processing capabilities
   - API integration limited to model providers
