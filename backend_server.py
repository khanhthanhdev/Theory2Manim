from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional
import asyncio
import os
import json
from generate_video import VideoGenerator
from mllm_tools.litellm import LiteLLMWrapper
from src.config.config import Config

# Load allowed models from JSON
allowed_models_path = os.path.join(os.path.dirname(__file__), 'src', 'utils', 'allowed_models.json')
with open(allowed_models_path, 'r') as f:
    allowed_models_data = json.load(f)
    allowed_models = allowed_models_data.get("allowed_models", [])

app = FastAPI()

class GenerateRequest(BaseModel):
    topic: str
    description: str
    model: str
    helper_model: Optional[str] = None
    output_dir: Optional[str] = Config.OUTPUT_DIR
    use_rag: Optional[bool] = False
    use_visual_fix_code: Optional[bool] = False
    use_context_learning: Optional[bool] = False
    max_retries: Optional[int] = 5
    max_scene_concurrency: Optional[int] = 1
    embedding_model: Optional[str] = Config.EMBEDDING_MODEL
    chroma_db_path: Optional[str] = Config.CHROMA_DB_PATH
    manim_docs_path: Optional[str] = Config.MANIM_DOCS_PATH
    context_learning_path: Optional[str] = Config.CONTEXT_LEARNING_PATH
    use_langfuse: Optional[bool] = False
    only_plan: Optional[bool] = False

class GenerateResponse(BaseModel):
    status: str
    message: str
    video_path: Optional[str] = None

@app.post("/generate", response_model=GenerateResponse)
async def generate_video_endpoint(request: GenerateRequest, background_tasks: BackgroundTasks):
    if request.model not in allowed_models:
        raise HTTPException(status_code=400, detail=f"Model {request.model} is not allowed.")
    if request.helper_model and request.helper_model not in allowed_models:
        raise HTTPException(status_code=400, detail=f"Helper model {request.helper_model} is not allowed.")

    # Prepare models
    planner_model = LiteLLMWrapper(
        model_name=request.model,
        temperature=0.7,
        print_cost=True,
        verbose=True,
        use_langfuse=request.use_langfuse
    )
    helper_model = LiteLLMWrapper(
        model_name=request.helper_model if request.helper_model else request.model,
        temperature=0.7,
        print_cost=True,
        verbose=True,
        use_langfuse=request.use_langfuse
    )
    scene_model = LiteLLMWrapper(
        model_name=request.model,
        temperature=0.7,
        print_cost=True,
        verbose=True,
        use_langfuse=request.use_langfuse
    )

    video_generator = VideoGenerator(
        planner_model=planner_model,
        scene_model=scene_model,
        helper_model=helper_model,
        output_dir=request.output_dir,
        verbose=True,
        use_rag=request.use_rag,
        use_context_learning=request.use_context_learning,
        context_learning_path=request.context_learning_path,
        chroma_db_path=request.chroma_db_path,
        manim_docs_path=request.manim_docs_path,
        embedding_model=request.embedding_model,
        use_visual_fix_code=request.use_visual_fix_code,
        use_langfuse=request.use_langfuse,
        max_scene_concurrency=request.max_scene_concurrency
    )

    async def process_and_combine():
        await video_generator.generate_video_pipeline(
            request.topic,
            request.description,
            max_retries=request.max_retries,
            only_plan=request.only_plan
        )
        # Combined video path
        file_prefix = request.topic.lower()
        import re
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', file_prefix)
        combined_path = os.path.join(request.output_dir, file_prefix, f"{file_prefix}_combined.mp4")
        return combined_path if os.path.exists(combined_path) else None

    # Background task wrapper for sync compatibility
    def background_process():
        asyncio.run(video_generator.generate_video_pipeline(
            request.topic,
            request.description,
            max_retries=request.max_retries,
            only_plan=request.only_plan
        ))

    # Add the background task
    background_tasks.add_task(background_process)
    # Immediately return response
    return GenerateResponse(status="processing", message="Video generation started. You can check the output directory later.")
